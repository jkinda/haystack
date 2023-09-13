from unittest.mock import patch
import pytest
import openai
from openai.util import convert_to_openai_object

import numpy as np


from haystack.preview.components.embedders.openai_text_embedder import OpenAITextEmbedder


def mock_openai_response(model: str = "text-embedding-ada-002", **kwargs) -> openai.openai_object.OpenAIObject:
    dict_response = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": np.random.rand(1536).tolist()}],
        "model": model,
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }

    return convert_to_openai_object(dict_response)


class TestOpenAITextEmbedder:
    @pytest.mark.unit
    def test_init_default(self):
        embedder = OpenAITextEmbedder(api_key="fake-api-key")

        assert embedder.api_key == "fake-api-key"
        assert embedder.model_name == "text-embedding-ada-002"
        assert embedder.api_base_url == "https://api.openai.com/v1"
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    @pytest.mark.unit
    def test_init_with_parameters(self):
        embedder = OpenAITextEmbedder(
            api_key="fake-api-key",
            model_name="model",
            api_base_url="https://custom-api-base-url.com",
            organization="fake-organization",
            prefix="prefix",
            suffix="suffix",
        )
        assert embedder.api_key == "fake-api-key"
        assert embedder.model_name == "model"
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.organization == "fake-organization"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"

    @pytest.mark.unit
    def test_to_dict(self):
        component = OpenAITextEmbedder(api_key="fake-api-key")
        data = component.to_dict()
        assert data == {
            "type": "OpenAITextEmbedder",
            "init_parameters": {
                "api_key": "fake-api-key",
                "model_name": "text-embedding-ada-002",
                "api_base_url": "https://api.openai.com/v1",
                "organization": None,
                "prefix": "",
                "suffix": "",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = OpenAITextEmbedder(
            api_key="fake-api-key",
            model_name="model",
            api_base_url="https://custom-api-base-url.com",
            organization="fake-organization",
            prefix="prefix",
            suffix="suffix",
        )
        data = component.to_dict()
        assert data == {
            "type": "OpenAITextEmbedder",
            "init_parameters": {
                "api_key": "fake-api-key",
                "model_name": "model",
                "api_base_url": "https://custom-api-base-url.com",
                "organization": "fake-organization",
                "prefix": "prefix",
                "suffix": "suffix",
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "OpenAITextEmbedder",
            "init_parameters": {
                "api_key": "fake-api-key",
                "model_name": "model",
                "api_base_url": "https://custom-api-base-url.com",
                "organization": "fake-organization",
                "prefix": "prefix",
                "suffix": "suffix",
            },
        }
        component = OpenAITextEmbedder.from_dict(data)
        assert component.api_key == "fake-api-key"
        assert component.model_name == "model"
        assert component.api_base_url == "https://custom-api-base-url.com"
        assert component.organization == "fake-organization"
        assert component.prefix == "prefix"
        assert component.suffix == "suffix"

    @pytest.mark.unit
    def test_run(self):
        model = "text-similarity-ada-001"

        with patch(
            "haystack.preview.components.embedders.openai_text_embedder.openai.Embedding"
        ) as openai_embedding_patch:
            openai_embedding_patch.create.side_effect = mock_openai_response

            embedder = OpenAITextEmbedder(api_key="fake-api-key", model_name=model, prefix="prefix ", suffix=" suffix")
            result = embedder.run(text="The food was delicious")

            openai_embedding_patch.create.assert_called_once_with(
                model=model, input="prefix The food was delicious suffix"
            )

        assert len(result["embedding"]) == 1536
        assert all([isinstance(x, float) for x in result["embedding"]])
        assert result["metadata"] == {"model": model, "usage": {"prompt_tokens": 4, "total_tokens": 4}}

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        embedder = OpenAITextEmbedder(api_key="fake-api-key")

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OpenAITextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)
