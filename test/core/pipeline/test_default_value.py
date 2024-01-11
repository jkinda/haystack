# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from haystack.core.component import component
from haystack.core.pipeline import Pipeline

import logging

logging.basicConfig(level=logging.DEBUG)


@component
class WithDefault:
    @component.output_types(c=int)
    def run(self, a: int, b: int = 2):
        return {"c": a + b}


def test_pipeline(tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("with_defaults", WithDefault())
    pipeline.draw(tmp_path / "default_value.png")

    # Pass all the inputs
    results = pipeline.run({"with_defaults": {"a": 40, "b": 30}})
    assert results == {"with_defaults": {"c": 70}}

    # Rely on default value for 'b'
    results = pipeline.run({"with_defaults": {"a": 40}})
    assert results == {"with_defaults": {"c": 42}}


def test_pipeline_wait_on_connected_optional_inputs(tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("first", WithDefault())
    pipeline.add_component("second", WithDefault())
    pipeline.add_component("third", WithDefault())
    pipeline.connect("first", "second.a")
    pipeline.connect("second", "third.b")

    pipeline.draw(tmp_path / "default_value.png")

    results = pipeline.run({"first": {"a": 1}, "third": {"a": 2}})
    assert results == {"third": {"c": 7}}
