# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Optional
from unittest.mock import patch

import pytest

from haystack.core.component.types import InputSocket, OutputSocket
from haystack.core.errors import PipelineDrawingError, PipelineError, PipelineRuntimeError
from haystack.core.pipeline import Pipeline
from haystack.testing.factory import component_class
from haystack.testing.sample_components import AddFixedValue, Double

logging.basicConfig(level=logging.DEBUG)


@patch("haystack.core.pipeline.pipeline._to_mermaid_image")
@patch("haystack.core.pipeline.pipeline.is_in_jupyter")
@patch("IPython.display.Image")
@patch("IPython.display.display")
def test_show_in_notebook(mock_ipython_display, mock_ipython_image, mock_is_in_jupyter, mock_to_mermaid_image):
    pipe = Pipeline()

    mock_to_mermaid_image.return_value = b"some_image_data"
    mock_is_in_jupyter.return_value = True

    pipe.show()
    mock_ipython_image.assert_called_once_with(b"some_image_data")
    mock_ipython_display.assert_called_once()


@patch("haystack.core.pipeline.pipeline.is_in_jupyter")
def test_show_not_in_notebook(mock_is_in_jupyter):
    pipe = Pipeline()

    mock_is_in_jupyter.return_value = False

    with pytest.raises(PipelineDrawingError):
        pipe.show()


@patch("haystack.core.pipeline.pipeline._to_mermaid_image")
def test_draw(mock_to_mermaid_image, tmp_path):
    pipe = Pipeline()
    mock_to_mermaid_image.return_value = b"some_image_data"

    image_path = tmp_path / "test.png"
    pipe.draw(path=image_path)
    assert image_path.read_bytes() == mock_to_mermaid_image.return_value


def test_add_component_to_different_pipelines():
    first_pipe = Pipeline()
    second_pipe = Pipeline()
    some_component = component_class("Some")()

    assert some_component.__haystack_added_to_pipeline__ is None
    first_pipe.add_component("some", some_component)
    assert some_component.__haystack_added_to_pipeline__ is first_pipe

    with pytest.raises(PipelineError):
        second_pipe.add_component("some", some_component)


def test_get_component_name():
    pipe = Pipeline()
    some_component = component_class("Some")()
    pipe.add_component("some", some_component)

    assert pipe.get_component_name(some_component) == "some"


def test_get_component_name_not_added_to_pipeline():
    pipe = Pipeline()
    some_component = component_class("Some")()

    assert pipe.get_component_name(some_component) == ""


@patch.object(Pipeline, "draw")
def test_repr(mock_draw):
    # Simulate not being in a notebook
    mock_draw.side_effect = ValueError

    pipe = Pipeline(metadata={"test": "test"}, max_loops_allowed=42)
    pipe.add_component("add_two", AddFixedValue(add=2))
    pipe.add_component("add_default", AddFixedValue())
    pipe.add_component("double", Double())
    pipe.connect("add_two", "double")
    pipe.connect("double", "add_default")

    expected_repr = (
        f"{object.__repr__(pipe)}\n"
        "🧱 Metadata\n"
        "  - test: test\n"
        "🚅 Components\n"
        "  - add_two: AddFixedValue\n"
        "  - add_default: AddFixedValue\n"
        "  - double: Double\n"
        "🛤️ Connections\n"
        "  - add_two.result -> double.value (int)\n"
        "  - double.value -> add_default.value (int)\n"
    )
    assert repr(pipe) == expected_repr

    mock_draw.assert_called_once_with()


def test_repr_on_notebook():
    pipe = Pipeline(metadata={"test": "test"}, max_loops_allowed=42)
    pipe.add_component("add_two", AddFixedValue(add=2))
    pipe.add_component("add_default", AddFixedValue())
    pipe.add_component("double", Double())
    pipe.connect("add_two", "double")
    pipe.connect("double", "add_default")

    with patch.object(Pipeline, "draw") as mock_draw:
        assert repr(pipe) == ""
        mock_draw.assert_called_once_with()


def test_run_with_component_that_does_not_return_dict():
    BrokenComponent = component_class(
        "BrokenComponent", input_types={"a": int}, output_types={"b": int}, output=1  # type:ignore
    )

    pipe = Pipeline(max_loops_allowed=10)
    pipe.add_component("comp", BrokenComponent())
    with pytest.raises(PipelineRuntimeError):
        pipe.run({"comp": {"a": 1}})


def test_to_dict():
    add_two = AddFixedValue(add=2)
    add_default = AddFixedValue()
    double = Double()
    pipe = Pipeline(metadata={"test": "test"}, max_loops_allowed=42)
    pipe.add_component("add_two", add_two)
    pipe.add_component("add_default", add_default)
    pipe.add_component("double", double)
    pipe.connect("add_two", "double")
    pipe.connect("double", "add_default")

    res = pipe.to_dict()
    expected = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 42,
        "components": {
            "add_two": {
                "type": "haystack.testing.sample_components.add_value.AddFixedValue",
                "init_parameters": {"add": 2},
            },
            "add_default": {
                "type": "haystack.testing.sample_components.add_value.AddFixedValue",
                "init_parameters": {"add": 1},
            },
            "double": {"type": "haystack.testing.sample_components.double.Double", "init_parameters": {}},
        },
        "connections": [
            {"sender": "add_two.result", "receiver": "double.value"},
            {"sender": "double.value", "receiver": "add_default.value"},
        ],
    }
    assert res == expected


def test_from_dict():
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 101,
        "components": {
            "add_two": {
                "type": "haystack.testing.sample_components.add_value.AddFixedValue",
                "init_parameters": {"add": 2},
            },
            "add_default": {
                "type": "haystack.testing.sample_components.add_value.AddFixedValue",
                "init_parameters": {"add": 1},
            },
            "double": {"type": "haystack.testing.sample_components.double.Double", "init_parameters": {}},
        },
        "connections": [
            {"sender": "add_two.result", "receiver": "double.value"},
            {"sender": "double.value", "receiver": "add_default.value"},
        ],
    }
    pipe = Pipeline.from_dict(data)

    assert pipe.metadata == {"test": "test"}
    assert pipe.max_loops_allowed == 101

    # Components
    assert len(pipe.graph.nodes) == 3
    ## add_two
    add_two = pipe.graph.nodes["add_two"]
    assert add_two["instance"].add == 2
    assert add_two["input_sockets"] == {
        "value": InputSocket(name="value", type=int),
        "add": InputSocket(name="add", type=Optional[int], default_value=None),
    }
    assert add_two["output_sockets"] == {"result": OutputSocket(name="result", type=int, receivers=["double"])}
    assert add_two["visits"] == 0

    ## add_default
    add_default = pipe.graph.nodes["add_default"]
    assert add_default["instance"].add == 1
    assert add_default["input_sockets"] == {
        "value": InputSocket(name="value", type=int, senders=["double"]),
        "add": InputSocket(name="add", type=Optional[int], default_value=None),
    }
    assert add_default["output_sockets"] == {"result": OutputSocket(name="result", type=int)}
    assert add_default["visits"] == 0

    ## double
    double = pipe.graph.nodes["double"]
    assert double["instance"]
    assert double["input_sockets"] == {"value": InputSocket(name="value", type=int, senders=["add_two"])}
    assert double["output_sockets"] == {"value": OutputSocket(name="value", type=int, receivers=["add_default"])}
    assert double["visits"] == 0

    # Connections
    connections = list(pipe.graph.edges(data=True))
    assert len(connections) == 2
    assert connections[0] == (
        "add_two",
        "double",
        {
            "conn_type": "int",
            "from_socket": OutputSocket(name="result", type=int, receivers=["double"]),
            "to_socket": InputSocket(name="value", type=int, senders=["add_two"]),
            "mandatory": True,
        },
    )
    assert connections[1] == (
        "double",
        "add_default",
        {
            "conn_type": "int",
            "from_socket": OutputSocket(name="value", type=int, receivers=["add_default"]),
            "to_socket": InputSocket(name="value", type=int, senders=["double"]),
            "mandatory": True,
        },
    )


def test_from_dict_with_empty_dict():
    assert Pipeline() == Pipeline.from_dict({})


def test_from_dict_with_components_instances():
    add_two = AddFixedValue(add=2)
    add_default = AddFixedValue()
    components = {"add_two": add_two, "add_default": add_default}
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {
            "add_two": {},
            "add_default": {},
            "double": {"type": "haystack.testing.sample_components.double.Double", "init_parameters": {}},
        },
        "connections": [
            {"sender": "add_two.result", "receiver": "double.value"},
            {"sender": "double.value", "receiver": "add_default.value"},
        ],
    }
    pipe = Pipeline.from_dict(data, components=components)
    assert pipe.metadata == {"test": "test"}
    assert pipe.max_loops_allowed == 100

    # Components
    assert len(pipe.graph.nodes) == 3
    ## add_two
    add_two_data = pipe.graph.nodes["add_two"]
    assert add_two_data["instance"] is add_two
    assert add_two_data["instance"].add == 2
    assert add_two_data["input_sockets"] == {
        "value": InputSocket(name="value", type=int),
        "add": InputSocket(name="add", type=Optional[int], default_value=None),
    }
    assert add_two_data["output_sockets"] == {"result": OutputSocket(name="result", type=int, receivers=["double"])}
    assert add_two_data["visits"] == 0

    ## add_default
    add_default_data = pipe.graph.nodes["add_default"]
    assert add_default_data["instance"] is add_default
    assert add_default_data["instance"].add == 1
    assert add_default_data["input_sockets"] == {
        "value": InputSocket(name="value", type=int, senders=["double"]),
        "add": InputSocket(name="add", type=Optional[int], default_value=None),
    }
    assert add_default_data["output_sockets"] == {"result": OutputSocket(name="result", type=int, receivers=[])}
    assert add_default_data["visits"] == 0

    ## double
    double = pipe.graph.nodes["double"]
    assert double["instance"]
    assert double["input_sockets"] == {"value": InputSocket(name="value", type=int, senders=["add_two"])}
    assert double["output_sockets"] == {"value": OutputSocket(name="value", type=int, receivers=["add_default"])}
    assert double["visits"] == 0

    # Connections
    connections = list(pipe.graph.edges(data=True))
    assert len(connections) == 2
    assert connections[0] == (
        "add_two",
        "double",
        {
            "conn_type": "int",
            "from_socket": OutputSocket(name="result", type=int, receivers=["double"]),
            "to_socket": InputSocket(name="value", type=int, senders=["add_two"]),
            "mandatory": True,
        },
    )
    assert connections[1] == (
        "double",
        "add_default",
        {
            "conn_type": "int",
            "from_socket": OutputSocket(name="value", type=int, receivers=["add_default"]),
            "to_socket": InputSocket(name="value", type=int, senders=["double"]),
            "mandatory": True,
        },
    )


def test_from_dict_without_component_type():
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {"add_two": {"init_parameters": {"add": 2}}},
        "connections": [],
    }
    with pytest.raises(PipelineError) as err:
        Pipeline.from_dict(data)

    err.match("Missing 'type' in component 'add_two'")


def test_from_dict_without_registered_component_type(request):
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {"add_two": {"type": "foo.bar.baz", "init_parameters": {"add": 2}}},
        "connections": [],
    }
    with pytest.raises(PipelineError) as err:
        Pipeline.from_dict(data)

    err.match(r"Component .+ not imported.")


def test_from_dict_without_connection_sender():
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {},
        "connections": [{"receiver": "some.receiver"}],
    }
    with pytest.raises(PipelineError) as err:
        Pipeline.from_dict(data)

    err.match("Missing sender in connection: {'receiver': 'some.receiver'}")


def test_from_dict_without_connection_receiver():
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {},
        "connections": [{"sender": "some.sender"}],
    }
    with pytest.raises(PipelineError) as err:
        Pipeline.from_dict(data)

    err.match("Missing receiver in connection: {'sender': 'some.sender'}")


def test_falsy_connection():
    A = component_class("A", input_types={"x": int}, output={"y": 0})
    B = component_class("A", input_types={"x": int}, output={"y": 0})
    p = Pipeline()
    p.add_component("a", A())
    p.add_component("b", B())
    p.connect("a.y", "b.x")
    assert p.run({"a": {"x": 10}})["b"]["y"] == 0


def test_describe_input_only_no_inputs_components():
    A = component_class("A", input_types={}, output={"x": 0})
    B = component_class("B", input_types={}, output={"y": 0})
    C = component_class("C", input_types={"x": int, "y": int}, output={"z": 0})
    p = Pipeline()
    p.add_component("a", A())
    p.add_component("b", B())
    p.add_component("c", C())
    p.connect("a.x", "c.x")
    p.connect("b.y", "c.y")
    assert p.inputs() == {}


def test_describe_input_some_components_with_no_inputs():
    A = component_class("A", input_types={}, output={"x": 0})
    B = component_class("B", input_types={"y": int}, output={"y": 0})
    C = component_class("C", input_types={"x": int, "y": int}, output={"z": 0})
    p = Pipeline()
    p.add_component("a", A())
    p.add_component("b", B())
    p.add_component("c", C())
    p.connect("a.x", "c.x")
    p.connect("b.y", "c.y")
    assert p.inputs() == {"b": {"y": {"type": int, "is_mandatory": True}}}


def test_describe_input_all_components_have_inputs():
    A = component_class("A", input_types={"x": Optional[int]}, output={"x": 0})
    B = component_class("B", input_types={"y": int}, output={"y": 0})
    C = component_class("C", input_types={"x": int, "y": int}, output={"z": 0})
    p = Pipeline()
    p.add_component("a", A())
    p.add_component("b", B())
    p.add_component("c", C())
    p.connect("a.x", "c.x")
    p.connect("b.y", "c.y")
    assert p.inputs() == {
        "a": {"x": {"type": Optional[int], "is_mandatory": True}},
        "b": {"y": {"type": int, "is_mandatory": True}},
    }


def test_describe_output_multiple_possible():
    """
    This pipeline has two outputs:
    {"b": {"output_b": {"type": str}}, "a": {"output_a": {"type": str}}}
    """
    A = component_class("A", input_types={"input_a": str}, output={"output_a": "str", "output_b": "str"})
    B = component_class("B", input_types={"input_b": str}, output={"output_b": "str"})

    pipe = Pipeline()
    pipe.add_component("a", A())
    pipe.add_component("b", B())
    pipe.connect("a.output_b", "b.input_b")

    assert pipe.outputs() == {"b": {"output_b": {"type": str}}, "a": {"output_a": {"type": str}}}


def test_describe_output_single():
    """
    This pipeline has one output:
    {"c": {"z": {"type": int}}}
    """
    A = component_class("A", input_types={"x": Optional[int]}, output={"x": 0})
    B = component_class("B", input_types={"y": int}, output={"y": 0})
    C = component_class("C", input_types={"x": int, "y": int}, output={"z": 0})
    p = Pipeline()
    p.add_component("a", A())
    p.add_component("b", B())
    p.add_component("c", C())
    p.connect("a.x", "c.x")
    p.connect("b.y", "c.y")

    assert p.outputs() == {"c": {"z": {"type": int}}}


def test_describe_no_outputs():
    """
    This pipeline sets up elaborate connections between three components but in fact it has no outputs:
    Check that p.outputs() == {}
    """
    A = component_class("A", input_types={"x": Optional[int]}, output={"x": 0})
    B = component_class("B", input_types={"y": int}, output={"y": 0})
    C = component_class("C", input_types={"x": int, "y": int}, output={})
    p = Pipeline()
    p.add_component("a", A())
    p.add_component("b", B())
    p.add_component("c", C())
    p.connect("a.x", "c.x")
    p.connect("b.y", "c.y")
    assert p.outputs() == {}
