from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.components import AddValue, Subtract

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):

    add_two = AddValue(add=2)
    diff = Subtract()

    pipeline = Pipeline()
    pipeline.add_component("first_addition", add_two)
    pipeline.add_component("second_addition", add_two)
    pipeline.add_component("third_addition", add_two)
    pipeline.add_component("diff", diff)
    pipeline.add_component("fourth_addition", AddValue(add=1))

    pipeline.connect("first_addition", "second_addition")
    pipeline.connect("second_addition", "diff.first_value")
    pipeline.connect("third_addition", "diff.second_value")
    pipeline.connect("diff", "fourth_addition")

    try:
        pipeline.draw(tmp_path / "merging_pipeline.png")
    except ImportError:
        logging.warning("pygraphviz not found, pipeline is not being drawn.")

    results = pipeline.run(
        {
            "first_addition": {"value": 1},
            "third_addition": {"value": 1},
        }
    )
    pprint(results)

    assert results == {"fourth_addition": {"value": 3}}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
