"""shell.py — read the `SHELL = {...}` literal out of a `Room.py`.

`Room.py` is room_ops' artifact, so parsing it belongs here rather than inside a consumer. It
lived under `agent/tools/check_collisions/source/geometry.py` while collision QC was its only
caller; `room_ops/glb_meta.py` now needs it too, and `room_ops` may not import `agent`
(see tests/test_architecture.py). `geometry._extract_shell` re-exports this, so every existing
caller keeps working unchanged.
"""

from __future__ import annotations

import ast
import json
import re


def extract_shell(src: str) -> dict:
    """Brace-matched parse of the `SHELL = {...}` literal — robust to whatever code follows it
    (the shell-editing pass moves the old `\\n\\nif __name__` anchor a regex relied on).

    JSON is the fast path, but the authoring/QC MODEL edits `SHELL` as Python, and the moment it
    writes something valid-Python-but-not-JSON — an adjacent-string note (`"a" "b"`), a trailing
    comma, a tuple — `json.loads` fails and the OLD code returned `{}`, silently blinding every
    geometry check below (no walls, no objects → every clash "passes"). `ast.literal_eval` parses
    the same Python literal, so fall back to it before giving up."""
    m = re.search(r"\bSHELL\s*=\s*\{", src)
    if not m:
        return {}
    i = src.index("{", m.start())
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                blob = src[i:j + 1]
                for parse in (json.loads, ast.literal_eval):
                    try:
                        return parse(blob)
                    except Exception:  # noqa: BLE001
                        continue
                return {}
    return {}
