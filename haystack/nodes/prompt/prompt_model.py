import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union, overload

from haystack.nodes.base import BaseComponent
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer
from haystack.schema import Document, MultiLabel
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_import:
    import torch


logger = logging.getLogger(__name__)


class PromptModel(BaseComponent):
    """
    The PromptModel class is a component that uses a pre-trained model to perform tasks defined in a prompt. Out of
    the box, it supports model invocation layers for:
    - Hugging Face transformers (all text2text-generation and text-generation models)
    - OpenAI InstructGPT models
    - Azure OpenAI InstructGPT models

    Although it's possible to use PromptModel to make prompt invocations on the underlying model, use
    PromptNode to interact with the model. PromptModel instances are a way for multiple
    PromptNode instances to use a single PromptNode, and thus save computational resources.

    For more details, refer to [PromptModels](https://docs.haystack.deepset.ai/docs/prompt_node#models).
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-base",
        max_length: Optional[int] = 100,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: Optional[bool] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        invocation_layer_class: Optional[Type[PromptModelInvocationLayer]] = None,
        model_kwargs: Optional[Dict] = None,
    ):
        """
        Creates an instance of PromptModel.

        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum number of tokens the output text generated by the model can have.
        :param api_key: The API key to use for the model.
        :param use_auth_token: The Hugging Face token to use.
        :param use_gpu: Whether to use GPU or not.
        :param devices: The devices to use where the model is loaded.
        :param invocation_layer_class: The custom invocation layer class to use. If None, known invocation layers are used.
        :param model_kwargs: Additional keyword arguments passed to the underlying model.

        Note that Azure OpenAI InstructGPT models require two additional parameters: azure_base_url (The URL for the
        Azure OpenAI API endpoint, usually in the form `https://<your-endpoint>.openai.azure.com') and
        azure_deployment_name (the name of the Azure OpenAI API deployment). You should add these parameters
        in the `model_kwargs` dictionary.
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.api_key = api_key
        self.timeout = timeout
        self.use_auth_token = use_auth_token
        self.use_gpu = use_gpu
        self.devices = devices

        self.model_kwargs = model_kwargs if model_kwargs else {}
        self.model_invocation_layer = self.create_invocation_layer(invocation_layer_class=invocation_layer_class)

    def create_invocation_layer(
        self, invocation_layer_class: Optional[Type[PromptModelInvocationLayer]]
    ) -> PromptModelInvocationLayer:
        kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "use_auth_token": self.use_auth_token,
            "use_gpu": self.use_gpu,
            "devices": self.devices,
        }
        all_kwargs = {**self.model_kwargs, **kwargs}

        if invocation_layer_class:
            return invocation_layer_class(
                model_name_or_path=self.model_name_or_path, max_length=self.max_length, **all_kwargs
            )

        for invocation_layer in PromptModelInvocationLayer.invocation_layer_providers:
            if inspect.isabstract(invocation_layer):
                continue
            if invocation_layer.supports(self.model_name_or_path, **all_kwargs):
                return invocation_layer(
                    model_name_or_path=self.model_name_or_path, max_length=self.max_length, **all_kwargs
                )
        raise ValueError(
            f"Model {self.model_name_or_path} is not supported - no matching invocation layer found."
            f" Currently supported invocation layers are: {PromptModelInvocationLayer.invocation_layer_providers}"
            f" You can implement and provide custom invocation layer for {self.model_name_or_path} by subclassing "
            "PromptModelInvocationLayer."
        )

    def invoke(self, prompt: Union[str, List[str], List[Dict[str, str]]], **kwargs) -> List[str]:
        """
        Takes in a prompt and returns a list of responses using the underlying invocation layer.

        :param prompt: The prompt to use for the invocation. It can be a single prompt or a list of prompts.
        :param kwargs: Additional keyword arguments to pass to the invocation layer.
        :return: A list of model-generated responses for the prompt or prompts.
        """
        output = self.model_invocation_layer.invoke(prompt=prompt, **kwargs)
        return output

    @overload
    def _ensure_token_limit(self, prompt: str) -> str:
        ...

    @overload
    def _ensure_token_limit(self, prompt: List[Dict[str, str]]) -> List[Dict[str, str]]:
        ...

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Ensure that length of the prompt and answer is within the maximum token length of the PromptModel.

        :param prompt: Prompt text to be sent to the generative model.
        """
        return self.model_invocation_layer._ensure_token_limit(prompt=prompt)

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
    ) -> Tuple[Dict, str]:
        raise NotImplementedError("This method should never be implemented in the derived class")

    def run_batch(
        self,
        queries: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        raise NotImplementedError("This method should never be implemented in the derived class")

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.__dict__)
