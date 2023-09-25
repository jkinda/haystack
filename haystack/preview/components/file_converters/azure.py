import json
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

from haystack.preview.lazy_imports import LazyImport
from haystack.preview import component, Document, default_to_dict, default_from_dict


with LazyImport(
    message="Run 'pip install azure-ai-formrecognizer>=3.2.0b2'"
) as azure_import:
    from azure.ai.formrecognizer import DocumentAnalysisClient, AnalyzeResult
    from azure.core.credentials import AzureKeyCredential


@component
class AzureOCRDocumentConverter:
    """
    A component for converting files to Documents using Azure's Document Intelligence service.
    Supported file formats are: PDF, JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, and HTML.

    In order to be able to use this component, you need an active Azure account
    and a Document Intelligence or Cognitive Services resource. Please follow the steps described in the
    [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api)
    to set up your resource.
    """

    def __init__(
        self,
        endpoint: str,
        credential_key: str,
        model_id: str = "prebuilt-read",
        save_json: bool = False,
        id_hash_keys: Optional[List[str]] = None,
    ):
        """
        Create an AzureOCRDocumentConverter component.

        :param endpoint: The endpoint of your Azure resource.
        :param credential_key: The key of your Azure resource.
        :param model_id: The model ID of the model you want to use. Please refer to [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/choose-model-feature)
            for a list of available models. Default: `"prebuilt-read"`.
        :param save_json: Whether to save the JSON output of the Azure API. Default: `False`.
        :param id_hash_keys: Generate the Document ID from a custom list of strings that refer to the Document's
            attributes. If you want to ensure you don't have duplicate Documents in your Document Store but texts are not
            unique, you can pass the name of the metadata to use when building the document ID (like
            `["text", "category"]`) to this field. In this case, the ID will be generated by using the text and the content of the
            `category` field. Default: `None`.
        """
        azure_import.check()

        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(credential_key)
        )
        self.endpoint = endpoint
        self.credential_key = credential_key
        self.model_id = model_id
        self.save_json = save_json
        self.id_hash_keys = id_hash_keys or []

    @component.output_types(documents=List[Document])
    def run(
        self, paths: List[Union[str, Path]], save_json: Optional[bool] = None, id_hash_keys: Optional[List[str]] = None
    ):
        """
        Convert files to Documents using Azure's Document Intelligence service.

        :param paths: Paths to the files to convert.
        :param save_json: Whether to save the JSON output of the Azure API. If not set, the value passed to the
            constructor will be used. Default: `None`.
        :param id_hash_keys: Generate the Document ID from a custom list of strings that refer to the Document's
            attributes. If you want to ensure you don't have duplicate Documents in your Document Store but texts are not
            unique, you can pass the name of the metadata to use when building the document ID (like
            `["text", "category"]`) to this field. In this case, the ID will be generated by using the text and the
            content of the `category` field.
            If not set, the id_hash_keys passed to the constructor will be used.
            Default: `None`.
        """
        id_hash_keys = id_hash_keys or self.id_hash_keys
        save_json = save_json if save_json is not None else self.save_json

        documents = []
        for path in paths:
            path = Path(path)
            with open(path, "rb") as file:
                poller = self.document_analysis_client.begin_analyze_document(model_id=self.model_id, document=file)
                result = poller.result()

            if save_json:
                with open(path.with_suffix(".json"), "w") as json_file:
                    json.dump(result.to_dict(), json_file, indent=2)

            file_suffix = path.suffix
            document = AzureOCRDocumentConverter._convert_azure_result_to_document(result, id_hash_keys, file_suffix)
            documents.append(document)

        return {"documents": documents}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            endpoint=self.endpoint,
            credential_key=self.credential_key,
            model_id=self.model_id,
            save_json=self.save_json,
            id_hash_keys=self.id_hash_keys,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzureOCRDocumentConverter":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    @staticmethod
    def _convert_azure_result_to_document(result: AnalyzeResult, id_hash_keys: List[str], file_suffix: str) -> Document:
        """
        Convert the result of Azure OCR to a Haystack text Document.
        """
        if file_suffix == ".pdf":
            text = ""
            for page in result.pages:
                lines = page.lines if page.lines else []
                for line in lines:
                    text += f"{line.content}\n"

                text += "\f"
        else:
            text = result.content

        if id_hash_keys:
            document = Document(text=text, id_hash_keys=id_hash_keys)
        else:
            document = Document(text=text)

        return document
