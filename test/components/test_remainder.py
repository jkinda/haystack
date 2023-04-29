from dataclasses import dataclass, make_dataclass
import pytest

from canals import component


@component
class Remainder:
    """
    Redirects the value, unchanged, along the connection corresponding to the remainder
    of a division. For example, if `divisor=3`, the value `5` would be sent along
    the second output connection.

    Single input, multi output decision component. Order of output connections is critical.
    """

    @dataclass
    class Output:
        odd: int

    def __init__(self, divisor: int = 2):
        self.divisor = divisor
        setattr(
            self,
            f"Output_{divisor}",
            make_dataclass("Output", [(f"remainder_is_{val}", int) for val in range(divisor)]),
        )

    def _output_types(self):
        return

    def run(self, value: int) -> None:  # Output:
        """
        :param divisor: the number to divide the input value for.
        :param input: the name of the input connection.
        :param outputs: the name of the output connections. Must be equal in length to the
            divisor (if dividing by 3, you must give exactly three output names).
            Ordering is important.
        """
        remainder = value % self.divisor
        # output = Remainder.Output()
        # setattr(output, str(remainder), remainder)
        return None  # output


# def test_remainder_default():
#     component = Remainder()
#     results = component.run(name="test_component", data=[("value", 10)], parameters={})
#     assert results == ({"0": 10}, {})

#     results = component.run(name="test_component", data=[("value", 11)], parameters={})
#     assert results == ({"1": 11}, {})
#     assert component.init_parameters == {}


# def test_remainder_default_output_for_divisor():
#     component = Remainder(divisor=5)
#     results = component.run(name="test_component", data=[("value", 10)], parameters={})
#     assert results == ({"0": 10}, {})

#     results = component.run(name="test_component", data=[("value", 13)], parameters={})
#     assert results == ({"3": 13}, {})
#     assert component.init_parameters == {"divisor": 5}


# def test_remainder_init_params():
#     with pytest.raises(ValueError):
#         component = Remainder(divisor=3, input="test", outputs=["one", "two"])

#     with pytest.raises(ValueError):
#         component = Remainder(divisor=3, input="test", outputs=["zero", "one", "two", "three"])

#     component = Remainder(divisor=3, input="test", outputs=["zero", "one", "two"])
#     results = component.run(name="test_component", data=[("value", 10)], parameters={})
#     assert results == ({"one": 10}, {})
#     assert component.init_parameters == {"divisor": 3, "input": "test", "outputs": ["zero", "one", "two"]}
