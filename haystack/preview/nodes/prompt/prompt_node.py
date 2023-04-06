import logging
from typing import Dict, List, Optional, Tuple, Any, Literal

from jinja2 import Template as JinjaTemplate, Environment as JinjaEnvironment, meta as JinjaMeta

from haystack.preview import node
from haystack.preview.nodes.prompt.providers.base import get_model


logger = logging.getLogger(__name__)


PromptTemplates = Literal[
    "question-answering",
    "question-generation",
    "conditioned-question-generation",
    "summarization",
    "question-answering-check",
    "sentiment-analysis",
    "multiple-choice-question-answering",
    "topic-classification",
    "language-detection",
    "translation",
    "zero-shot-react",
]


PROVIDER_MODULES = ["haystack.preview"]


PREDEFINED_TEMPLATES = {
    "question-answering": "Given the context please answer the question. Context: {{ documents }}; Question: {{ question }}; Answer:",
    "question-generation": "Given the context please generate a question. Context: {{ documents }}; Question:",
    "conditioned-question-generation": "Please come up with a question for the given context and the answer. Context: {{ documents }}; Answer: $answers; Question:",
    "summarization": "Summarize this document: {{ documents }} Summary:",
    "question-answering-check": "Does the following context contain the answer to the question? Context: {{ documents }}; Question: {{ question }}s; Please answer yes or no! Answer:",
    "sentiment-analysis": "Please give a sentiment for this context. Answer with positive, negative or neutral. Context: {{ documents }}; Answer:",
    "multiple-choice-question-answering": "Question:{{ question }}s ; Choose the most suitable option to answer the above question. Options: $options; Answer:",
    "topic-classification": "Categories: {{ options }}; What category best describes: {{ documents }}; Answer:",
    "language-detection": "Detect the language in the following context and answer with the name of the language. Context: {{ documents }}; Answer:",
    "translation": "Translate the following context to {{ target_language }}. Context: {{ documents }}; Translation:",
    "zero-shot-react": """
You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions
correctly, you have access to the following tools:

{{ tool_names_with_descriptions }}

To answer questions, you'll need to go through multiple steps involving step-by-step thinking and
selecting appropriate tools and their inputs; tools will respond with observations. When you are ready
for a final answer, respond with the `Final Answer:`

Use the following format:

Question: the question to be answered
Thought: Reason if you have the final answer. If yes, answer the question. If not, find out the missing information needed to answer it.
Tool: pick one of $tool_names
Tool Input: the input for the tool
Observation: the tool will respond with the result
...
Final Answer: the final answer to the question, make it short (1-5 words)

Thought, Tool, Tool Input, and Observation steps can be repeated multiple times, but sometimes we can find an answer in the first pass
---

Question: {{ question }}
Thought: Let's think step-by-step, I first need to """,
}


@node
class PromptNode:
    """
    The PromptNode class is the central abstraction in Haystack's large language model (LLM) support. PromptNode
    supports multiple NLP tasks out of the box. You can use it to perform tasks, such as
    summarization, question answering, question generation, and more, using a single, unified model within the Haystack
    framework.

    One of the benefits of PromptNode is that you can use it to define and add additional prompt templates
    the model supports. Defining additional prompt templates makes it possible to extend the model's capabilities
    and use it for a broader range of NLP tasks in Haystack. Prompt engineers define templates
    for each NLP task and register them with PromptNode. The burden of defining templates for each task rests on
    the prompt engineers, not the users.

    Using an instance of the PromptModel class, you can create multiple PromptNodes that share the same model, saving
    the memory and time required to load the model multiple times.

    PromptNode also supports multiple model invocation layers:
    - Hugging Face transformers (all text2text-generation models)
    - OpenAI InstructGPT models
    - Azure OpenAI InstructGPT models

    We recommend using LLMs fine-tuned on a collection of datasets phrased as instructions, otherwise we find that the
    LLM does not "follow" prompt instructions well. This is why we recommend using T5 flan or OpenAI InstructGPT models.

    For more details, see [PromptNode](https://docs.haystack.deepset.ai/docs/prompt_node).
    """

    def __init__(
        self,
        template: Optional[PromptTemplates] = None,
        custom_template: Optional[str] = None,
        output: Optional[str] = "answers",
        model_name_or_path: str = "google/flan-t5-base",
        model_provider: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        provider_modules: Optional[List[str]] = None,
    ):
        """
        Creates a PromptNode instance.

        NOTE: the variables that the template accepts must match with the variables you list in the `inputs` variable of
        this PromptNode. For example, if your prompt expects `{{ documents }}` and `{{ documents }}`, the inputs need to
        be `['question', 'documents']`. Inputs are automatically inferred from the template text.

        :param outputs: the outputs that this PromptNode will generate.
        :param template: The template of the prompt to use for the model. Must be the name of a predefined prompt
            template. To provide a custom template, use `custom_template`.
        :param custom_template: A custom template of the prompt to use for the model.
            To use Haystack's pre-defined templates, use `template`.
        :param model_name: The name of the model to use, like a HF model identifier or an OpenAI model name.
        :param model_provider: force a specific provider for the model. If not given, Haystack will find a provider for
            your model automatically.
        :param model_params: Parameters to be passed to the model provider, like API keys, init parameters, etc.
        :param provider_modules: if you have external model providers, add the module where they are, like
            `haystack.preview`
        """
        self.jinja_env = JinjaEnvironment()
        if template:
            self.template = PREDEFINED_TEMPLATES[template]
        elif custom_template:
            self.template = custom_template
        else:
            raise ValueError("Provide either a template or a custom_template for this PromptNode.")

        self.inputs = self.find_inputs()
        self.outputs = [output]
        self.model = None
        self.model_name_or_path = model_name_or_path
        self.model_provider = model_provider
        self.model_params = model_params
        self.provider_modules = provider_modules or PROVIDER_MODULES
        self.init_parameters = {
            "output": output,
            "template": template,
            "custom_template": custom_template,
            "model_name_or_path": model_name_or_path,
            "model_kwargs": model_params,
            "model_provider": model_provider,
        }

    def warm_up(self):
        if not self.model:
            self.model = get_model(
                model_name_or_path=self.model_name_or_path,
                model_provider=self.model_provider,
                model_params=self.model_params,
                modules_to_search=self.provider_modules,
            )

    def run(
        self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        It takes in a prompt, and returns a list of responses using the underlying invocation layer.

        :param prompt: The prompt to use for the invocation. It can be a single prompt or a list of prompts.
        :param kwargs: Additional keyword arguments to pass to the invocation layer.
        :return: A list of model generated responses for the prompt or prompts.
        """
        data = {key: value for key, value in data}
        params = parameters.get(name, {})
        model_params = params.pop("model_params", {})
        output = self.prompt(validate_inputs=False, model_params=model_params, **data, **params)
        return ({self.outputs[0]: output}, parameters)

    def prompt(
        self, validate_inputs: bool = True, model_params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[str]:
        """
        It takes in a prompt, and returns a list of responses using the underlying model.

        :param prompt: The prompt to use for the invocation.
        :param kwargs: Additional keyword arguments to pass to the model.
        :return: A list of model generated responses for the prompt.
        """
        self.warm_up()
        if validate_inputs:
            self.validate_inputs(given_inputs=list(kwargs.keys()))
        prompt = JinjaTemplate(self.template).render(**kwargs)
        return self.model.invoke(prompt=prompt, model_params=model_params or {})

    def find_inputs(self) -> List[str]:
        """
        Find out which inputs this template requires.

        :param template: the template to infer the required variables from.
        :returns: a list of strings corresponding to the variables found in the template.
        """
        parsed_content = self.jinja_env.parse(self.template)
        return JinjaMeta.find_undeclared_variables(parsed_content)

    def validate_inputs(self, given_inputs: List[str]) -> None:
        """
        Make sure the template variables and the input variables match.

        :param given_inputs: the inputs to validate
        :raises ValueError if the template is using variables that are not found
            in the inputs list or vice versa.
        """
        if sorted(self.inputs) != sorted(given_inputs):
            raise ValueError(
                "The values given to this PromptNode do not match the variables it expects:\n- Expected variables: "
                f"{self.inputs}\n- Given variables: {given_inputs}\nConnect this node to another node that outputs "
                "these variables, pass this value to `PromptNode.prompt()`, or change template."
            )
