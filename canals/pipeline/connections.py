# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, Optional, List, Any, get_args

import logging
import itertools

from canals.errors import PipelineConnectError
from canals.pipeline.sockets import InputSocket, OutputSocket


logger = logging.getLogger(__name__)


def parse_connection_name(connection: str) -> Tuple[str, Optional[str]]:
    """
    Returns component-connection pairs from a connect_to/from string
    """
    if "." in connection:
        split_str = connection.split(".", maxsplit=1)
        return (split_str[0], split_str[1])
    return connection, None


def find_unambiguous_connection(
    from_node: str, to_node: str, from_sockets: List[OutputSocket], to_sockets: List[InputSocket]
) -> Tuple[OutputSocket, InputSocket]:
    """
    Find one single possible connection between two lists of sockets.
    """
    # List all combinations of sockets that match by type
    possible_connections = [
        (out_sock, in_sock)
        for out_sock, in_sock in itertools.product(from_sockets, to_sockets)
        if not in_sock.sender and (Any in in_sock.types or out_sock.types == in_sock.types)
    ]

    # No connections seem to be possible
    if not possible_connections:
        connections_status_str = _connections_status(
            from_node=from_node, from_sockets=from_sockets, to_node=to_node, to_sockets=to_sockets
        )
        raise PipelineConnectError(
            f"Cannot connect '{from_node}' with '{to_node}': "
            f"no matching connections available.\n{connections_status_str}"
        )

    # There's more than one possible connection
    if len(possible_connections) > 1:
        # Try to match by name
        name_matches = [
            (out_sock, in_sock) for out_sock, in_sock in possible_connections if in_sock.name == out_sock.name
        ]
        if len(name_matches) != 1:
            # TODO allow for multiple connections at once if there is no ambiguity?
            # TODO give priority to sockets that have no default values?
            connections_status_str = _connections_status(
                from_node=from_node, from_sockets=from_sockets, to_node=to_node, to_sockets=to_sockets
            )
            raise PipelineConnectError(
                f"Cannot connect '{from_node}' with '{to_node}': more than one connection is possible "
                "between these components. Please specify the connection name, like: "
                f"pipeline.connect('{from_node}.{possible_connections[0][0].name}', "
                f"'{to_node}.{possible_connections[0][1].name}').\n{connections_status_str}"
            )

    return possible_connections[0]


def _connections_status(from_node: str, to_node: str, from_sockets: List[OutputSocket], to_sockets: List[InputSocket]):
    """
    Lists the status of the sockets, for error messages.
    """
    from_sockets_entries = []
    for from_socket in from_sockets:
        socket_types = ", ".join([_get_socket_type_desc(t) for t in from_socket.types])
        from_sockets_entries.append(f" - {from_socket.name} ({socket_types})")
    from_sockets_list = "\n".join(from_sockets_entries)

    to_sockets_entries = []
    for to_socket in to_sockets:
        socket_types = ", ".join([_get_socket_type_desc(t) for t in to_socket.types])
        to_sockets_entries.append(
            f" - {to_socket.name} ({socket_types}), {'sent by '+to_socket.sender if to_socket.sender else 'available'}"
        )
    to_sockets_list = "\n".join(to_sockets_entries)

    return f"'{from_node}':\n{from_sockets_list}\n'{to_node}':\n{to_sockets_list}"


def _get_socket_type_desc(type_):
    """
    Assembles a readable representation of the type of a connection. Can handle primitive types, classes, and
    arbitrarily nested structures of types from the typing module.
    """
    # get_args returns something only if this type has subtypes, in which case it needs to be printed differently.
    args = get_args(type_)

    if not args:
        if isinstance(type_, type):
            return type_.__name__
        # Literals only accept instances, not classes, so we need to account for those.
        return str(type_) if not isinstance(type_, str) else f"'{type_}'"  # Quote strings

    # Python < 3.10 support
    if not hasattr(type_, "__name__"):
        return str(type_)

    subtypes = ", ".join([_get_socket_type_desc(subtype) for subtype in args if subtype is not type(None)])
    return f"{type_.__name__}[{subtypes}]"
