#!/usr/bin/env python3
"""make_selfcontained.py — turn a blender_lib-based build recipe into ONE
self-contained Blender script (`object.py`) that has no external imports.

The pipeline authors build recipes that do `from blender_lib import ...`, which
only run when blender_lib.py is importable (an absolute sys.path hack is fragile
and breaks the moment the file moves). This bundler inlines the library straight
into the recipe, so the result *is* the object's definition and runs anywhere:

    blender -b --python object.py                       # uses ./textures, writes ./<name>.glb
    blender -b --python object.py -- <texdir> <out.glb>

It is a pure text transform (no Blender needed). It:
  * keeps the recipe's docstring as the file header (+ a self-contained note),
  * inlines blender_lib.py's body (minus its docstring/imports),
  * drops the recipe's `import`/`sys.path.insert`/`from blender_lib import ...`
    lines and its TEXDIR/OUT_GLB arg block, replacing them with a portable,
    `__file__`-relative arg block.

Usage:
    python3 make_selfcontained.py <recipe.py> <out_object.py> [--name NAME] [--lib blender_lib.py]
"""

import argparse
import ast
import os
import re

HELP_HEADER = "# ==== inlined Blender helpers (generated from blender_lib.py; do not edit) ===="
EDIT_HEADER = "# ==== OBJECT DEFINITION — edit below to change the object ===="

# stdlib / Blender modules we re-add in a single unified import line
DROP_IMPORTS = {"bpy", "bmesh", "math", "mathutils", "os", "sys"}


def strip_module_docstring(src):
    """Return (docstring_or_None, source_without_its_docstring)."""
    tree = ast.parse(src)
    doc = ast.get_docstring(tree, clean=False)
    if doc is None or not tree.body:
        return doc, src
    node = tree.body[0]
    lines = src.splitlines(keepends=True)
    del lines[node.lineno - 1 : node.end_lineno]
    return doc, "".join(lines)


def _droppable(line, drop_assigns):
    s = line.strip()
    if not s:
        return False
    m = re.match(r"^import\s+([\w.]+)(\s+as\s+\w+)?\s*$", s)
    if m and m.group(1).split(".")[0] in DROP_IMPORTS:
        return True
    if s.startswith("sys.path.insert") or s.startswith("import blender_lib"):
        return True
    return any(re.match(rf"^{re.escape(name)}\s*=", s) for name in drop_assigns)


def drop_lines(src, drop_assigns=()):
    """Drop blender_lib imports, sys.path hacks and given top-level assignments.
    Handles the multi-line `from blender_lib import (...)` form."""
    out, lines, i = [], src.splitlines(keepends=True), 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("from blender_lib import"):
            if "(" in s and ")" not in s:  # multi-line import block
                while i < len(lines) and ")" not in lines[i]:
                    i += 1
            i += 1
            continue
        if _droppable(lines[i], drop_assigns):
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)


def _path_vars(src):
    """Names used as `sys.path.insert(0, NAME)` — their assignments are now dead
    (e.g. `SKILL = "/abs/.../scripts"`) and should be dropped too."""
    return set(re.findall(r"sys\.path\.insert\(\s*\d+\s*,\s*([A-Za-z_]\w*)\s*\)", src))


def bundle(recipe_path, lib_path, name):
    recipe_raw = open(recipe_path).read()
    rdoc, recipe_body = strip_module_docstring(recipe_raw)
    recipe_body = drop_lines(
        recipe_body, ("argv", "_argv", "TEXDIR", "OUT_GLB", *_path_vars(recipe_raw))
    )
    _, lib_body = strip_module_docstring(open(lib_path).read())
    lib_body = drop_lines(lib_body)

    doc_lines = (rdoc or f"Self-contained Blender build script for {name}.").rstrip().splitlines()
    for idx, ln in enumerate(
        doc_lines
    ):  # drop the recipe's stale "Run:" block; we re-add a correct one
        if ln.strip().lower().startswith("run:"):
            doc_lines = doc_lines[:idx]
            break
    doc = "\n".join(doc_lines).rstrip()
    header = (
        f'"""{doc}\n\n'
        "SELF-CONTAINED — the Blender helper library is inlined below, so this file\n"
        "defines the object on its own (no external imports / sys.path hacks).\n\n"
        "Run:\n"
        f"  blender -b --python object.py                       # uses ./textures, writes ./{name}.glb\n"
        "  blender -b --python object.py -- <texdir> <out.glb>\n"
        '"""\n'
    )
    args_block = (
        "import bpy, bmesh, math, mathutils, os, sys\n\n"
        'HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()\n'
        "_argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []\n"
        "TEXDIR = _argv[0] if _argv else os.path.join(HERE, 'textures')\n"
        f"OUT_GLB = _argv[1] if len(_argv) > 1 else os.path.join(HERE, '{name}.glb')\n"
    )
    return "".join(
        [
            header,
            "\n",
            args_block,
            "\n",
            HELP_HEADER,
            "\n",
            lib_body.strip("\n"),
            "\n\n",
            EDIT_HEADER,
            "\n",
            recipe_body.strip("\n"),
            "\n",
        ]
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("recipe", help="build recipe that uses `from blender_lib import ...`")
    ap.add_argument("out", help="output path for the self-contained script (e.g. object.py)")
    ap.add_argument("--name", default=None, help="object name for the default OUT_GLB filename")
    ap.add_argument(
        "--lib", default=None, help="blender_lib.py path (default: next to this script)"
    )
    a = ap.parse_args()
    lib = a.lib or os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_lib.py")
    name = a.name or os.path.splitext(os.path.basename(a.out))[0]
    open(a.out, "w").write(bundle(a.recipe, lib, name))
    print(f"wrote {a.out}  (name={name}, lib={lib})")


if __name__ == "__main__":
    main()
