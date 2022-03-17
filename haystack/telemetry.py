# coding: utf-8
"""
    Telemetry
    Haystack reports anonymous usage statistics to support continuous software improvements for all its users.
    An example report can be inspected via calling print_telemetry_report(). Check out the documentation for more details: https://haystack.deepset.ai/guides/telemetry
    You can opt-out of sharing usage statistics by setting the environment variable HAYSTACK_TELEMETRY_ENABLED to "False" or calling disable_telemetry().
    You can log all events to the local file specified in LOG_PATH for inspection by setting the environment variable HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED to "True".
"""
import sys
import uuid
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

import logging
import haystack
import transformers
import torch
import posthog
import os
import platform
from threading import Thread

posthog.api_key = "phc_uZ6c2kaqSz3u9QrYRCBbxsi6MMHiiew9dwi4LTobgV3"
posthog.host = "https://posthog.dpst.dev"
HAYSTACK_TELEMETRY_ENABLED = "HAYSTACK_TELEMETRY_ENABLED"
HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED = "HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED"
HAYSTACK_EXECUTION_CONTEXT = "HAYSTACK_EXECUTION_CONTEXT"
HAYSTACK_DOCKER_CONTAINER = "HAYSTACK_DOCKER_CONTAINER"
CONFIG_PATH = Path("~/.haystack/config.yaml").expanduser()
LOG_PATH = Path("~/.haystack/telemetry.log").expanduser()

telemetry_meta_data: Dict[str, Any] = {}
user_id: Optional[str] = None

logger = logging.getLogger(__name__)


class TelemetryFileType(Enum):
    LOG_FILE: str = "LOG_FILE"
    CONFIG_FILE: str = "CONFIG_FILE"


def print_telemetry_report():
    """
    Prints the user id and the meta data that are sent in events
    """
    if is_telemetry_enabled():
        user_id = _get_or_create_user_id()
        meta_data = _get_or_create_telemetry_meta_data()
        print({**{"user_id": user_id}, **meta_data})
    else:
        print("Telemetry is disabled.")


def enable_telemetry():
    """
    Enables telemetry so that a limited amount of anonymous usage data is sent as events.
    """
    os.environ[HAYSTACK_TELEMETRY_ENABLED] = "True"
    logger.info("Telemetry has been enabled.")


def disable_telemetry():
    """
    Disables telemetry so that no events are sent anymore, except for one final event.
    """
    os.environ[HAYSTACK_TELEMETRY_ENABLED] = "False"
    logger.info("Telemetry has been disabled.")


def enable_writing_events_to_file():
    """
    Enables writing each event that is sent to the log file specified in LOG_PATH
    """
    os.environ[HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED] = "True"
    logger.info(f"Writing events to log file {LOG_PATH} has been enabled.")


def disable_writing_events_to_file():
    """
    Disables writing each event that is sent to the log file specified in LOG_PATH
    """
    os.environ[HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED] = "False"
    logger.info(f"Writing events to log file {LOG_PATH} has been disabled.")


def is_telemetry_enabled() -> bool:
    """
    Returns False if telemetry is disabled via an environment variable, otherwise True.
    """
    telemetry_environ = os.environ.get(HAYSTACK_TELEMETRY_ENABLED, "True")
    return telemetry_environ.lower() != "false"


def is_telemetry_logging_to_file_enabled() -> bool:
    """
    Returns False if logging telemetry events to a file is disabled via an environment variable, otherwise True.
    """
    telemetry_environ = os.environ.get(HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED, "False")
    return telemetry_environ.lower() != "false"


def send_event_if_public_demo(func):
    """
    Can be used as a decorator to send an event only if HAYSTACK_EXECUTION_CONTEXT is "public_demo"
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        exec_context = os.environ.get(HAYSTACK_EXECUTION_CONTEXT, "")
        if exec_context == "public_demo":
            send_custom_event(event="demo query executed", payload=kwargs)
        return func(*args, **kwargs)

    return wrapper


def send_event(func):
    """
    Can be used as a decorator to send an event formatted like 'Pipeline.eval executed'
    with additional parameters as defined in TrackedParameters ('add_isolated_node_eval') and
    metadata, such as os_version
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        send_custom_event(event=f"{type(args[0]).__name__}.{func.__name__} executed", payload=kwargs)
        return func(*args, **kwargs)

    return wrapper


def send_custom_event(event: str = "", payload: Dict[str, Any] = {}):
    """
    This method can be called directly from anywhere in Haystack to send an event.
    Enriches the given event with metadata and sends it to the posthog server if telemetry is enabled.
    If telemetry has just been disabled, a final event is sent and the config file and the log file are deleted

    :param event: Name of the event. Use a noun and a verb, e.g., "evaluation started", "component created"
    :param payload: A dictionary containing event meta data, e.g., parameter settings
    """
    global user_id  # pylint: disable=global-statement
    try:

        def request_task(payload: Dict[str, Any], delete_telemetry_files: bool = False):
            """
            Sends an event in a post request to a posthog server

            :param payload: A dictionary containing event meta data, e.g., parameter settings
            :param delete_telemetry_files: Whether to delete the config and log file after sending the request. Used when sending a finale event after disabling telemetry.
            """
            event_properties = {**(NonPrivateParameters.apply_filter(payload)), **_get_or_create_telemetry_meta_data()}
            if user_id is None:
                raise RuntimeError("User id was not initialized")
            try:
                posthog.capture(distinct_id=user_id, event=event, properties=event_properties)
            except Exception as e:
                logger.debug("Telemetry was not able to make a post request to posthog.", exc_info=e)
            if is_telemetry_enabled() and is_telemetry_logging_to_file_enabled():
                _write_event_to_telemetry_log_file(distinct_id=user_id, event=event, properties=event_properties)
            if delete_telemetry_files:
                _delete_telemetry_file(TelemetryFileType.CONFIG_FILE)
                _delete_telemetry_file(TelemetryFileType.LOG_FILE)

        def fire_and_forget(payload: Dict[str, Any], delete_telemetry_files: bool = False):
            """
            Starts a thread with the task to send an event in a post request to a posthog server

            :param payload: A dictionary containing event meta data, e.g., parameter settings
            :param delete_telemetry_files: Whether to delete the config and log file after sending the request. Used when sending a finale event after disabling telemetry.
            """
            Thread(target=request_task, args=(payload, delete_telemetry_files)).start()

        user_id = _get_or_create_user_id()
        if is_telemetry_enabled():
            fire_and_forget(payload=payload)
        elif CONFIG_PATH.exists():
            # if telemetry has just been disabled but the config file has not been deleted yet,
            # then send a final event instead of the triggered event and delete config file and log file afterward
            event = "telemetry disabled"
            fire_and_forget(payload={}, delete_telemetry_files=True)
        else:
            # return without sending any event, not even a final event
            return

    except Exception as e:
        logger.debug("Telemetry was not able to send an event.", exc_info=e)


def send_tutorial_event(url: str):
    """
    Can be called when a tutorial dataset is downloaded so that the dataset URL is used to identify the tutorial and send an event.

    :param url: URL of the dataset that is loaded in the tutorial.
    """
    dataset_url_to_tutorial = {
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip": "1",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/squad_small.json.zip": "2",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt3.zip": "3",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/faq_covidbert.csv.zip": "4",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/nq_dev_subset_v2.json.zip": "5",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt6.zip": "6",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/small_generator_dataset.csv.zip": "7",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/preprocessing_tutorial8.zip": "8",
        # "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz":"9",
        "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz": "9",
        "https://fandom-qa.s3-eu-west-1.amazonaws.com/saved_models/hp_v3.4.zip": "10",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt11.zip": "11",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt12.zip": "12",
        # Tutorial 13: no dataset available yet
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt14.zip": "14",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/ottqa_tables_sample.json.zip": "15",
        # "https://nlp.stanford.edu/data/glove.6B.zip": "16",
        "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/preprocessing_tutorial16.zip": "16",
    }
    send_custom_event(event=f"tutorial {dataset_url_to_tutorial.get(url, '?')} executed")


def _get_or_create_user_id() -> str:
    """
    Randomly generates a user id or loads the id defined in the config file and returns it.
    """
    global user_id  # pylint: disable=global-statement
    if user_id is None:
        # if user_id is not set, read it from config file
        _read_telemetry_config()
        if user_id is None:
            # if user_id cannot be read from config file, create new user_id and write it to config file
            user_id = str(uuid.uuid4())
            _write_telemetry_config()
    return user_id


def _get_or_create_telemetry_meta_data() -> Dict[str, Any]:
    """
    Collects meta data about the setup that is used with Haystack, such as: operating system, python version, Haystack version, transformers version, pytorch version, number of GPUs, execution environment, and the value stored in the env variable HAYSTACK_EXECUTION_CONTEXT.
    """
    global telemetry_meta_data  # pylint: disable=global-statement
    if not telemetry_meta_data:
        telemetry_meta_data = {
            "os_version": platform.release(),
            "os_family": platform.system(),
            "os_machine": platform.machine(),
            "python_version": platform.python_version(),
            "haystack_version": haystack.__version__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "n_gpu": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "context": os.environ.get(HAYSTACK_EXECUTION_CONTEXT),
            "execution_env": _get_execution_environment(),
        }
    return telemetry_meta_data


def _get_execution_environment():
    """
    Identifies the execution environment that Haystack is running in.
    Options are: colab notebook, kubernetes, CPU/GPU docker container, test environment, jupyter notebook, python script
    """
    if "google.colab" in sys.modules:
        execution_env = "colab"
    elif "KUBERNETES_SERVICE_HOST" in os.environ:
        execution_env = "kubernetes"
    elif HAYSTACK_DOCKER_CONTAINER in os.environ:
        execution_env = os.environ.get(HAYSTACK_DOCKER_CONTAINER)
    # check if pytest is imported
    elif "pytest" in sys.modules:
        execution_env = "test"
    else:
        try:
            shell = get_ipython().__class__.__name__  # pylint: disable=undefined-variable
            execution_env = shell
        except Exception:
            execution_env = "script"
    return execution_env


def _read_telemetry_config():
    """
    Loads the config from the file specified in CONFIG_PATH
    """
    global user_id  # pylint: disable=global-statement
    try:
        if not CONFIG_PATH.is_file():
            return
        with open(CONFIG_PATH, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
            if "user_id" in config and user_id is None:
                user_id = config["user_id"]
    except Exception as e:
        logger.debug(f"Telemetry was not able to read the config file {CONFIG_PATH}.", exc_info=e)


def _write_telemetry_config():
    """
    Writes a config file storing the randomly generated user id and whether to write events to a log file.
    This method logs an info to inform the user about telemetry when it is used for the first time.
    """
    global user_id  # pylint: disable=global-statement
    try:
        # show a log message if telemetry config is written for the first time
        if not CONFIG_PATH.is_file():
            logger.info(
                f'Haystack sends anonymous usage data to understand the actual usage and steer dev efforts towards features that are most meaningful to users. You can opt out at anytime by setting {HAYSTACK_TELEMETRY_ENABLED}="False" as an environment variable or by calling disable_telemetry(). More information at https://haystack.deepset.ai/guides/telemetry'
            )
            CONFIG_PATH.parents[0].mkdir(parents=True, exist_ok=True)
        user_id = _get_or_create_user_id()
        config = {"user_id": user_id}

        with open(CONFIG_PATH, "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
    except Exception:
        logger.debug(f"Could not write config file to {CONFIG_PATH}.")
        send_custom_event(event="config saving failed")


def _write_event_to_telemetry_log_file(distinct_id: str, event: str, properties: Dict[str, Any]):
    try:
        with open(LOG_PATH, "a") as file_object:
            file_object.write(f"{event}, {properties}, {distinct_id}\n")
    except Exception as e:
        logger.debug(f"Telemetry was not able to write event to log file {LOG_PATH}.", exc_info=e)


def _delete_telemetry_file(file_type_to_delete: TelemetryFileType):
    """
    Deletes the telemetry config file or log file if it exists.
    """
    if not isinstance(file_type_to_delete, TelemetryFileType):
        logger.debug("File type to delete must be either TelemetryFileType.LOG_FILE or TelemetryFileType.CONFIG_FILE.")
    path = LOG_PATH if file_type_to_delete is TelemetryFileType.LOG_FILE else CONFIG_PATH
    try:
        path.unlink(missing_ok=True)
    except Exception as e:
        logger.debug(f"Telemetry was not able to delete the {file_type_to_delete} at {path}.", exc_info=e)


class NonPrivateParameters:
    param_names: List[str] = ["top_k", "model_name_or_path", "add_isolated_node_eval"]

    @classmethod
    def apply_filter(cls, param_dicts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures that only the values of non-private parameters are sent in events. All other parameter values are filtered out before sending an event.
        If model_name_or_path is a local file path, it will be reduced to the name of the file. The directory names are not sent.

        :param param_dicts: the keyword arguments that need to be filtered before sending an event
        """
        tracked_params = {k: param_dicts[k] for k in cls.param_names if k in param_dicts}

        # if model_name_or_path is a local file path, we reduce it to the model name
        if "model_name_or_path" in tracked_params:
            if (
                Path(tracked_params["model_name_or_path"]).is_file()
                or tracked_params["model_name_or_path"].count(os.path.sep) > 1
            ):
                # if model_name_or_path points to an existing file or contains more than one / it is a path
                tracked_params["model_name_or_path"] = Path(tracked_params["model_name_or_path"]).name
        return tracked_params
