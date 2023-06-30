# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

import networkx

# from networkx.drawing.nx_agraph import to_agraph as nx_to_agraph


logger = logging.getLogger(__name__)


def to_agraph(graph: networkx.MultiDiGraph):
    """
    Renders a pipeline graph using PyGraphViz. You need to install it and all its system dependencies for it to work.
    """
    try:
        import pygraphviz  # pylint: disable=unused-import,import-outside-toplevel
        from networkx.drawing.nx_agraph import (  # pylint: disable=unused-import,import-outside-toplevel
            to_agraph as nx_to_agraph,
        )

        for inp, outp, key, data in graph.out_edges("input", keys=True, data=True):
            data["style"] = "dashed"
            graph.add_edge(inp, outp, key=key, **data)

        for inp, outp, key, data in graph.in_edges("output", keys=True, data=True):
            data["style"] = "dashed"
            graph.add_edge(inp, outp, key=key, **data)

        graph.nodes["input"]["shape"] = "plain"
        graph.nodes["output"]["shape"] = "plain"
        agraph = nx_to_agraph(graph)
        agraph.layout("dot")
        return agraph

    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError(
            "Can't use 'pygraphviz' to draw this pipeline: pygraphviz could not be imported. "
            "Make sure pygraphviz is installed and all its system dependencies are setup correctly."
        ) from exc
