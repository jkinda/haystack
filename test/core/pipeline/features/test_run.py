from pytest_bdd import scenarios, given
import pytest

from haystack import Pipeline, component
from haystack.components.others import Multiplexer
from haystack.core.errors import PipelineMaxLoops
from haystack.testing.sample_components import (
    Accumulate,
    AddFixedValue,
    Double,
    Greet,
    Parity,
    Repeat,
    Subtract,
    Sum,
    Threshold,
    Remainder,
)
from haystack.testing.factory import component_class


pytestmark = pytest.mark.integration

scenarios("pipeline_run.feature")


@given("a pipeline that has no components", target_fixture="pipeline_data")
def pipeline_that_has_no_components():
    pipeline = Pipeline()
    inputs = {}
    expected_outputs = {}
    return pipeline, inputs, expected_outputs, []


@given("a pipeline that is linear", target_fixture="pipeline_data")
def pipeline_that_is_linear():
    pipeline = Pipeline()
    pipeline.add_component("first_addition", AddFixedValue(add=2))
    pipeline.add_component("second_addition", AddFixedValue())
    pipeline.add_component("double", Double())
    pipeline.connect("first_addition", "double")
    pipeline.connect("double", "second_addition")

    return (
        pipeline,
        {"first_addition": {"value": 1}},
        {"second_addition": {"result": 7}},
        ["first_addition", "double", "second_addition"],
    )


@given("a pipeline that has an infinite loop", target_fixture="pipeline_data")
def pipeline_that_has_an_infinite_loop():
    def custom_init(self):
        component.set_input_type(self, "x", int)
        component.set_input_type(self, "y", int, 1)
        component.set_output_types(self, a=int, b=int)

    FakeComponent = component_class("FakeComponent", output={"a": 1, "b": 1}, extra_fields={"__init__": custom_init})
    pipe = Pipeline(max_loops_allowed=1)
    pipe.add_component("first", FakeComponent())
    pipe.add_component("second", FakeComponent())
    pipe.connect("first.a", "second.x")
    pipe.connect("second.b", "first.y")
    return pipe, {"first": {"x": 1}}, PipelineMaxLoops


@given("a pipeline that is really complex with lots of components, forks, and loops", target_fixture="pipeline_data")
def pipeline_complex():
    pipeline = Pipeline(max_loops_allowed=2)
    pipeline.add_component("greet_first", Greet(message="Hello, the value is {value}."))
    pipeline.add_component("accumulate_1", Accumulate())
    pipeline.add_component("add_two", AddFixedValue(add=2))
    pipeline.add_component("parity", Parity())
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("accumulate_2", Accumulate())

    pipeline.add_component("multiplexer", Multiplexer(type_=int))
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("double", Double())

    pipeline.add_component("greet_again", Greet(message="Hello again, now the value is {value}."))
    pipeline.add_component("sum", Sum())

    pipeline.add_component("greet_enumerator", Greet(message="Hello from enumerator, here the value became {value}."))
    pipeline.add_component("enumerate", Repeat(outputs=["first", "second"]))
    pipeline.add_component("add_three", AddFixedValue(add=3))

    pipeline.add_component("diff", Subtract())
    pipeline.add_component("greet_one_last_time", Greet(message="Bye bye! The value here is {value}!"))
    pipeline.add_component("replicate", Repeat(outputs=["first", "second"]))
    pipeline.add_component("add_five", AddFixedValue(add=5))
    pipeline.add_component("add_four", AddFixedValue(add=4))
    pipeline.add_component("accumulate_3", Accumulate())

    pipeline.connect("greet_first", "accumulate_1")
    pipeline.connect("accumulate_1", "add_two")
    pipeline.connect("add_two", "parity")

    pipeline.connect("parity.even", "greet_again")
    pipeline.connect("greet_again", "sum.values")
    pipeline.connect("sum", "diff.first_value")
    pipeline.connect("diff", "greet_one_last_time")
    pipeline.connect("greet_one_last_time", "replicate")
    pipeline.connect("replicate.first", "add_five.value")
    pipeline.connect("replicate.second", "add_four.value")
    pipeline.connect("add_four", "accumulate_3")

    pipeline.connect("parity.odd", "add_one.value")
    pipeline.connect("add_one", "multiplexer.value")
    pipeline.connect("multiplexer", "below_10")

    pipeline.connect("below_10.below", "double")
    pipeline.connect("double", "multiplexer.value")

    pipeline.connect("below_10.above", "accumulate_2")
    pipeline.connect("accumulate_2", "diff.second_value")

    pipeline.connect("greet_enumerator", "enumerate")
    pipeline.connect("enumerate.second", "sum.values")

    pipeline.connect("enumerate.first", "add_three.value")
    pipeline.connect("add_three", "sum.values")

    return (
        pipeline,
        {"greet_first": {"value": 1}, "greet_enumerator": {"value": 1}},
        {"accumulate_3": {"value": -7}, "add_five": {"result": -6}},
        [
            "greet_first",
            "accumulate_1",
            "add_two",
            "parity",
            "add_one",
            "multiplexer",
            "below_10",
            "double",
            "multiplexer",
            "below_10",
            "double",
            "multiplexer",
            "below_10",
            "accumulate_2",
            "greet_enumerator",
            "enumerate",
            "add_three",
            "sum",
            "diff",
            "greet_one_last_time",
            "replicate",
            "add_five",
            "add_four",
            "accumulate_3",
        ],
    )


@given("a pipeline that has a single component with a default input", target_fixture="pipeline_data")
def pipeline_that_has_a_single_component_with_a_default_input():
    @component
    class WithDefault:
        @component.output_types(b=int)
        def run(self, a: int, b: int = 2):
            return {"c": a + b}

    pipeline = Pipeline()
    pipeline.add_component("with_defaults", WithDefault())

    return (
        pipeline,
        [{"with_defaults": {"a": 40, "b": 30}}, {"with_defaults": {"a": 40}}],
        [{"with_defaults": {"c": 70}}, {"with_defaults": {"c": 42}}],
        [["with_defaults"], ["with_defaults"]],
    )


@given("a pipeline that has two loops of identical lengths", target_fixture="pipeline_data")
def pipeline_that_has_two_loops_of_identical_lengths():
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("multiplexer", Multiplexer(type_=int))
    pipeline.add_component("remainder", Remainder(divisor=3))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("multiplexer.value", "remainder.value")
    pipeline.connect("remainder.remainder_is_1", "add_two.value")
    pipeline.connect("remainder.remainder_is_2", "add_one.value")
    pipeline.connect("add_two", "multiplexer.value")
    pipeline.connect("add_one", "multiplexer.value")

    return (
        pipeline,
        [
            {"multiplexer": {"value": 0}},
            {"multiplexer": {"value": 3}},
            {"multiplexer": {"value": 4}},
            {"multiplexer": {"value": 5}},
            {"multiplexer": {"value": 6}},
        ],
        [
            {"remainder": {"remainder_is_0": 0}},
            {"remainder": {"remainder_is_0": 3}},
            {"remainder": {"remainder_is_0": 6}},
            {"remainder": {"remainder_is_0": 6}},
            {"remainder": {"remainder_is_0": 6}},
        ],
        [
            ["multiplexer", "remainder"],
            ["multiplexer", "remainder"],
            ["multiplexer", "remainder", "add_two", "multiplexer", "remainder"],
            ["multiplexer", "remainder", "add_one", "multiplexer", "remainder"],
            ["multiplexer", "remainder"],
        ],
    )


@given("a pipeline that has two loops of different lengths", target_fixture="pipeline_data")
def pipeline_that_has_two_loops_of_different_lengths():
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("multiplexer", Multiplexer(type_=int))
    pipeline.add_component("remainder", Remainder(divisor=3))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("add_two_1", AddFixedValue(add=1))
    pipeline.add_component("add_two_2", AddFixedValue(add=1))

    pipeline.connect("multiplexer.value", "remainder.value")
    pipeline.connect("remainder.remainder_is_1", "add_two_1.value")
    pipeline.connect("add_two_1", "add_two_2.value")
    pipeline.connect("add_two_2", "multiplexer")
    pipeline.connect("remainder.remainder_is_2", "add_one.value")
    pipeline.connect("add_one", "multiplexer")

    return (
        pipeline,
        [
            {"multiplexer": {"value": 0}},
            {"multiplexer": {"value": 3}},
            {"multiplexer": {"value": 4}},
            {"multiplexer": {"value": 5}},
            {"multiplexer": {"value": 6}},
        ],
        [
            {"remainder": {"remainder_is_0": 0}},
            {"remainder": {"remainder_is_0": 3}},
            {"remainder": {"remainder_is_0": 6}},
            {"remainder": {"remainder_is_0": 6}},
            {"remainder": {"remainder_is_0": 6}},
        ],
        [
            ["multiplexer", "remainder"],
            ["multiplexer", "remainder"],
            ["multiplexer", "remainder", "add_two_1", "add_two_2", "multiplexer", "remainder"],
            ["multiplexer", "remainder", "add_one", "multiplexer", "remainder"],
            ["multiplexer", "remainder"],
        ],
    )
