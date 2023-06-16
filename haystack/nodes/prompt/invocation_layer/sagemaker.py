import json
import os
from typing import Optional, Dict, Union, List, Any, Callable
import logging

import requests
import sseclient

from haystack.environment import HAYSTACK_REMOTE_API_TIMEOUT_SEC, HAYSTACK_REMOTE_API_MAX_RETRIES
from haystack.errors import (
    HuggingFaceInferenceLimitError,
    HuggingFaceInferenceUnauthorizedError,
    HuggingFaceInferenceError,
)
from haystack.nodes.prompt.invocation_layer import (
    PromptModelInvocationLayer,
    TokenStreamingHandler,
    DefaultTokenStreamingHandler,
)
from haystack.nodes.prompt.invocation_layer.handlers import DefaultPromptHandler
from haystack.utils import request_with_retry
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)
HF_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
HF_RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))

with LazyImport() as transformers_import:
    from transformers.pipelines import get_task


class SageMakerInvocationLayer(PromptModelInvocationLayer):
    """
    TODO

    """

    def __init__(self, model_name_or_path: str, **kwargs):
        """
        TODO
        """

        super().__init__(model_name_or_path)
        # TODO add authentication check + error msg

        # if not valid_api_key:
        #     raise ValueError(
        #         f"api_key {api_key} must be a valid Hugging Face token. "
        #         f"Your token is available in your Hugging Face settings page."
        #     )
        self.max_length = max_length

        # for a list of supported parameters (TODO: verify if all of them are supported in sagemaker)
        self.model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "best_of",
                "details",
                "do_sample",
                "max_new_tokens",
                "max_time",
                "model_max_length",
                "num_return_sequences",
                "repetition_penalty",
                "return_full_text",
                "seed",
                "stream",
                "stream_handler",
                "temperature",
                "top_k",
                "top_p",
                "truncate",
                "typical_p",
                "watermark",
            ]
            if key in kwargs
        }

        # we pop the model_max_length from the model_input_kwargs as it is not sent to the model
        # but used to truncate the prompt if needed
        model_max_length = self.model_input_kwargs.pop("model_max_length", 1024)

        # Truncate prompt if prompt tokens > model_max_length-max_length (max_lengt is the length of the generated text)
        self.prompt_handler = DefaultPromptHandler(
            model_name_or_path="gpt2", model_max_length=model_max_length, max_length=self.max_length or 100
        )

    def invoke(self, *args, **kwargs):
        """
        Invokes a prompt on the model. It takes in a prompt and returns a list of responses using a REST invocation.
        :return: The responses are being returned.
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                f"No prompt provided. Model {self.model_name_or_path} requires prompt."
                f"Make sure to provide prompt in kwargs."
            )

        # TODO: Verify that stop words are respected by SageMaker. If not, cut response off at first stop word occurence.
        stop_words = kwargs.pop("stop_words", None) or []
        kwargs_with_defaults = self.model_input_kwargs

        # TODO: Check if still relevant / needed
        if "max_new_tokens" not in kwargs_with_defaults:
            kwargs_with_defaults["max_new_tokens"] = self.max_length
        kwargs_with_defaults.update(kwargs)

        # TODO
        # either stream is True (will use default handler) or stream_handler is provided
        # stream = (
        #     kwargs_with_defaults.get("stream", False) or kwargs_with_defaults.get("stream_handler", None) is not None
        # )

        # see https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
        params = {
            "best_of": kwargs_with_defaults.get("best_of", None),
            "details": kwargs_with_defaults.get("details", True),
            "do_sample": kwargs_with_defaults.get("do_sample", False),
            "max_new_tokens": kwargs_with_defaults.get("max_new_tokens", self.max_length),
            "max_time": kwargs_with_defaults.get("max_time", None),
            "num_return_sequences": kwargs_with_defaults.get("num_return_sequences", None),
            "repetition_penalty": kwargs_with_defaults.get("repetition_penalty", None),
            "return_full_text": kwargs_with_defaults.get("return_full_text", False),
            "seed": kwargs_with_defaults.get("seed", None),
            "stop": kwargs_with_defaults.get("stop", stop_words),
            "temperature": kwargs_with_defaults.get("temperature", None),
            "top_k": kwargs_with_defaults.get("top_k", None),
            "top_p": kwargs_with_defaults.get("top_p", None),
            "truncate": kwargs_with_defaults.get("truncate", None),
            "typical_p": kwargs_with_defaults.get("typical_p", None),
            "watermark": kwargs_with_defaults.get("watermark", False),
        }
        # TODO: Change to boto request "invoke_endpoint"
        response: requests.Response = self._post(
            data={"inputs": prompt, "parameters": params, "stream": stream}, stream=stream
        )

        # if stream:
        #     handler: TokenStreamingHandler = kwargs_with_defaults.pop("stream_handler", DefaultTokenStreamingHandler())
        #     generated_texts = self._process_streaming_response(response, handler, stop_words)
        # else:
        output = json.loads(response.text)
        generated_texts = [o["generated_text"] for o in output if "generated_text" in o]
        return generated_texts

    # def _process_streaming_response(
    #     self, response: requests.Response, stream_handler: TokenStreamingHandler, stop_words: List[str]
    # ) -> List[str]:
    #     """
    #     Stream the response and invoke the stream_handler on each token.
    #
    #     :param response: The response object from the server.
    #     :param stream_handler: The handler to invoke on each token.
    #     :param stop_words: The stop words to ignore.
    #     """
    #     client = sseclient.SSEClient(response)
    #     tokens: List[str] = []
    #     try:
    #         for event in client.events():
    #             if event.data != TokenStreamingHandler.DONE_MARKER:
    #                 event_data = json.loads(event.data)
    #                 token: Optional[str] = self._extract_token(event_data)
    #                 # if valid token and not a stop words (we don't want to return stop words)
    #                 if token and token.strip() not in stop_words:
    #                     tokens.append(stream_handler(token, event_data=event_data))
    #     finally:
    #         client.close()
    #     return ["".join(tokens)]  # return a list of strings just like non-streaming

    # def _extract_token(self, event_data: Dict[str, Any]) -> Optional[str]:
    #     """
    #     Extract the token from the event data. If the token is a special token, return None.
    #     param event_data: Event data from the streaming response.
    #     """
    #     # extract token from event data and only consider non-special tokens
    #     return event_data["token"]["text"] if not event_data["token"]["special"] else None

    def _post(
        self,
        data: Dict[str, Any],
        stream: bool = False,
        attempts: int = HF_RETRIES,
        status_codes_to_retry: Optional[List[int]] = None,
        timeout: float = HF_TIMEOUT,
    ) -> requests.Response:
        """
        Post data to the HF inference model. It takes in a prompt and returns a list of responses using a REST invocation.
        :param data: The data to be sent to the model.
        :param stream: Whether to stream the response.
        :param attempts: The number of attempts to make.
        :param status_codes_to_retry: The status codes to retry on.
        :param timeout: The timeout for the request.
        :return: The responses are being returned.
        """
        response: requests.Response
        if status_codes_to_retry is None:
            status_codes_to_retry = [429]
        try:
            # TODO CHANGE TO BOTO REQUEST
            pass
            # response = request_with_retry(
            #     method="POST",
            #     status_codes_to_retry=status_codes_to_retry,
            #     attempts=attempts,
            #     url=self.url,
            #     headers=self.headers,
            #     json=data,
            #     timeout=timeout,
            #     stream=stream,
            # )
        except requests.HTTPError as err:
            res = err.response
            if res.status_code == 429:
                raise HuggingFaceInferenceLimitError(f"API rate limit exceeded: {res.text}")
            if res.status_code == 401:
                raise HuggingFaceInferenceUnauthorizedError(f"API key is invalid: {res.text}")

            raise HuggingFaceInferenceError(
                f"HuggingFace Inference returned an error.\nStatus code: {res.status_code}\nResponse body: {res.text}",
                status_code=res.status_code,
            )
        return response

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        # the prompt for this model will be of the type str
        resize_info = self.prompt_handler(prompt)  # type: ignore
        if resize_info["prompt_length"] != resize_info["new_prompt_length"]:
            logger.warning(
                "The prompt has been truncated from %s tokens to %s tokens so that the prompt length and "
                "answer length (%s tokens) fit within the max token limit (%s tokens). "
                "Shorten the prompt to prevent it from being cut off.",
                resize_info["prompt_length"],
                max(0, resize_info["model_max_length"] - resize_info["max_length"]),  # type: ignore
                resize_info["max_length"],
                resize_info["model_max_length"],
            )
        return str(resize_info["resized_prompt"])

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        # TODO check via boto3 if sagemaker endpoint exists (maybe also addition kwargs give us a hint, e.g. region)
        return True
