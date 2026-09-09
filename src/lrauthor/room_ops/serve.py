"""Binding a local port for the room viewers.

Shared by the walkable viewer and the live authoring viewer, which are otherwise unrelated: both
are started by hand, repeatedly, often while an older one is still up.
"""

from __future__ import annotations

import errno
from http.server import ThreadingHTTPServer

PORT_TRIES = 20


def bind(handler, host: str, port: int) -> ThreadingHTTPServer:
    """Bind the first free port at or above `port`.

    A viewer is very often started while an older one is still up — the previous run's, or this
    run's in another terminal — and the port a viewer happens to sit on is incidental to what was
    asked for. Refusing to start over it means looking at a room dies at the door for a reason that
    has nothing to do with the room, so walk up until one binds and let the caller announce where it
    landed rather than making someone pick a free number by hand.
    """
    for candidate in range(port, port + PORT_TRIES):
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    raise SystemExit(f"no free port in {port}-{port + PORT_TRIES - 1} on {host}")
