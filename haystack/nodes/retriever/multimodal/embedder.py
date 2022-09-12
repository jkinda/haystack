from typing import get_args, Union, Optional, Dict, List, Any

import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np
from PIL import Image

from haystack.modeling.model.multimodal import get_model
from haystack.errors import NodeError, ModelingError
from haystack.modeling.model.multimodal.base import HaystackModel
from haystack.schema import ContentTypes, Document


logger = logging.getLogger(__name__)


class MultiModalRetrieverError(NodeError):
    pass


FilterType = Dict[str, Union[Dict[str, Any], List[Any], str, int, float, bool]]


DOCUMENT_CONVERTERS = {
    # NOTE: Keep this '?' cleaning step, it needs to be double-checked for impact on the inference results.
    "text": lambda doc: doc.content[:-1] if doc.content[-1] == "?" else doc.content,
    "table": lambda doc: " ".join(
        doc.content.columns.tolist() + [cell for row in doc.content.values.tolist() for cell in row]
    ),
    "image": lambda doc: Image.open(doc.content),
}

CAN_EMBED_META = ["text", "table"]


def get_devices(devices: Optional[List[Union[str, torch.device]]]) -> List[torch.device]:
    """
    Convert a list of device names into a list of Torch devices,
    depending on the system's configuration and hardware.
    """
    if devices is not None:
        return [torch.device(device) for device in devices]
    elif torch.cuda.is_available():
        return [torch.device(device) for device in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


class MultiModalEmbedder:
    def __init__(
        self,
        embedding_models: Dict[ContentTypes, Union[Path, str]],
        feature_extractors_params: Dict[str, Dict[str, Any]] = None,
        batch_size: int = 16,
        embed_meta_fields: List[str] = ["name"],
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        """
        Init the Retriever and all its models from a local or remote model checkpoint.
        The checkpoint format matches huggingface transformers' model format.

        :param embedding_models: Dictionary matching a local path or remote name of encoder checkpoint with
            the content type it should handle ("text", "table", "image", etc...).
            The format equals the one used by hugging-face transformers' modelhub models.
            Expected input format: `{'text': 'name_or_path_to_text_model', 'image': 'name_or_path_to_image_model', etc...}`
            Keep in mind that the models should output in the same embedding space for this retriever to work.
        :param batch_size: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
        :param embed_meta_fields: Concatenate the provided meta fields and text passage / image to a text pair that is
                                  then used to create the embedding.
                                  This is the approach used in the original paper and is likely to improve
                                  performance if your titles contain meaningful information for retrieval
                                  (topic, entities etc.).
        :param similarity_function: Which function to apply for calculating the similarity of query and passage embeddings during training.
                                    Options: `dot_product` (Default) or `cosine`
        :param global_loss_buffer_size: Buffer size for all_gather() in DDP.
                                        Increase if errors like "encoded data exceeds max_size ..." come up
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param devices: List of GPU (or CPU) devices, to limit inference to certain GPUs and not use all available ones
                        These strings will be converted into pytorch devices, so use the string notation described here:
                        https://pytorch.org/docs/simage/tensor_attributes.html?highlight=torch%20device#torch.torch.device
                        (e.g. ["cuda:0"]). Note: as multi-GPU training is currently not implemented for TableTextRetriever,
                        training will only use the first device provided in this list.
        :param use_auth_token:  API token used to download private models from Huggingface. If this parameter is set to `True`,
                                the local token will be used, which must be previously created via `transformer-cli login`.
                                Additional information can be found here https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        super().__init__()

        self.devices = get_devices(devices)
        if batch_size < len(self.devices):
            logger.warning("Batch size is lower than the number of devices. Not all GPUs will be utilized.")

        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.embed_meta_fields = embed_meta_fields

        feature_extractors_params = {
            content_type: {"max_length": 256, **(feature_extractors_params or {}).get(content_type, {})}
            for content_type in get_args(ContentTypes)
        }

        self.models: Dict[str, HaystackModel] = {}
        for content_type, embedding_model in embedding_models.items():
            self.models[content_type] = get_model(
                pretrained_model_name_or_path=embedding_model,
                content_type=content_type,
                devices=self.devices,
                autoconfig_kwargs={"use_auth_token": use_auth_token},
                model_kwargs={"use_auth_token": use_auth_token},
                feature_extractor_kwargs=feature_extractors_params[content_type],
            )

    def embed(self, documents: List[Document], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Create embeddings for a list of documents using the relevant encoder for their content type.

        :param documents: Documents to embed
        :return: Embeddings, one per document, in the form of a np.array
        """
        batch_size = batch_size if batch_size is not None else self.batch_size

        all_embeddings = []
        for batch_index in tqdm(
            iterable=range(0, len(documents), batch_size),
            unit=" Docs",
            desc=f"Create embeddings",
            position=1,
            leave=False,
            disable=not self.progress_bar,
        ):
            docs_batch = documents[batch_index : batch_index + batch_size]
            data_by_type = self._docs_to_data(documents=docs_batch)

            # Get output for each model
            outputs_by_type: Dict[ContentTypes, torch.Tensor] = {}
            for data_type, data in data_by_type.items():

                model = self.models.get(data_type)
                if not model:
                    raise ModelingError(
                        f"Some data of type {data_type} was passed, but no model capable of handling such data was "
                        f"initialized. Initialized models: {', '.join(self.models.keys())}"
                    )
                outputs_by_type[data_type] = self.models[data_type].encode(data=data, convert_to_tensor=True)

            # Check the output sizes
            embedding_sizes = [output.shape[-1] for output in outputs_by_type.values()]
            if not all(embedding_size == embedding_sizes[0] for embedding_size in embedding_sizes):
                raise ModelingError(
                    "Some of the models are using a different embedding size. They should all match. "
                    f"Embedding sizes by model: "
                    f"{ {name: output.shape[-1] for name, output in outputs_by_type.items()} }"
                )

            # Combine the outputs in a single matrix
            outputs = torch.stack(list(outputs_by_type.values()))
            embeddings = outputs.view(-1, embedding_sizes[0])
            embeddings = embeddings.cpu()
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings)

    def _docs_to_data(self, documents: List[Document]) -> Dict[ContentTypes, List[Any]]:
        """
        Extract the data to embed from each document and returns them classified by content type.

        :param documents: the documents to prepare fur multimodal embedding.
        :return: a dictionary containing one key for each content type, and a list of data extracted
            from each document, ready to be passed to the feature extractor (for example the content
            of a text document, a linearized table, a PIL image object, etc...)
        """
        docs_data: Dict[ContentTypes, List[Any]] = {key: [] for key in get_args(ContentTypes)}
        for doc in documents:
            try:
                document_converter = DOCUMENT_CONVERTERS[doc.content_type]
            except KeyError as e:
                raise MultiModalRetrieverError(
                    f"Unknown content type '{doc.content_type}'. Known types: {', '.join(get_args(ContentTypes))}"
                ) from e

            data = document_converter(doc)

            if self.embed_meta_fields and doc.content_type in CAN_EMBED_META:
                meta = json.dumps(doc.meta or {})
                data = (
                    f"{meta} {data}" if meta else data
                )  # FIXME meta & data used to be returned as a tuple: verify it still works as intended

            docs_data[doc.content_type].append(data)

        return {key: values for key, values in docs_data.items() if values}
