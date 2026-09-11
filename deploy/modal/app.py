"""The one Modal app for LiteReality's models.

    MODAL_PROFILE=huangzhening uv run modal deploy --env main deploy/modal/app.py

Functions: ``trellis`` (TRELLIS.2, deploy/modal/trellis) and ``dino`` (GroundingDINO +
DINOv2, deploy/modal/dino). A deploy replaces every function on the app, which is why
both are deployed together here and never one at a time.
"""

from __future__ import annotations

import modal
from dino.app import app as dino_app
from trellis.app import app as trellis_app

APP_NAME = "litereality"

app = modal.App(APP_NAME)
app.include(trellis_app)
app.include(dino_app)
