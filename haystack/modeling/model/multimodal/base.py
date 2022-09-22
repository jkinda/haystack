from typing import Any, List, Union, Optional

import logging
from pathlib import Path
from abc import ABC, abstractmethod

import torch

from haystack.schema import ContentTypes


logger = logging.getLogger(__name__)


class HaystackModel(ABC):
    """
    Interface on top of HaystackTransformer and HaystackSentenceTransformer
    """

    def __init__(self, pretrained_model_name_or_path: Union[str, Path], model_type: str, content_type: ContentTypes):
        """
        :param pretrained_model_name_or_path: name of the model to load
        :param model_type: the value of model_type from the model's Config
        :param content_type: the type of data (text, image, ...) the model is supposed to process.
            See the values of `haystack.schema.ContentTypes`.
        """
        logger.info(
            f" 🤖 Loading '{pretrained_model_name_or_path}' "
            f"({self.__class__.__name__} of type '{model_type if model_type else '<unknown>'}' "
            f"for {content_type} data)"
        )
        self.model_name_or_path = pretrained_model_name_or_path
        self.model_type = model_type
        self.content_type = content_type

    @abstractmethod
    def encode(self, data: List[Any], **kwargs) -> torch.Tensor:
        """
        Run the model on the input data to obtain output vectors.
        """
        raise NotImplementedError("Abstract method, use a subclass.")

    @abstractmethod
    def to(self, devices: Optional[List[torch.device]]) -> None:
        """
        Send the model to the specified PyTorch device(s)
        """
        raise NotImplementedError("Abstract method, use a subclass.")

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """
        The output embedding size.
        """
        raise NotImplementedError("Abstract method, use a subclass.")
