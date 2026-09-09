"""Sim-ready physical properties for generated articulated objects.

`properties` compiles a built GLB into the model; `urdf` and `mjcf` export it; `checks` proves it
with a solver. The model is the source of truth — the exports are pure functions of it.
"""

from .properties import SimModel, build_model, load_model, write_model

__all__ = ["SimModel", "build_model", "load_model", "write_model"]
