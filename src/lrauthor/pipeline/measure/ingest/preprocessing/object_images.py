"""Object image operations used by the ingest crop stage."""

from __future__ import annotations

import os

from .vendor.litereality.scene_preprocessing import (
    Colors,
    prepare_camera_data_for_retrieval,
    process_all_folders,
)
from .vendor.litereality.scene_preprocessing import (
    process_object_images as _process_object_images,
)

__all__ = [
    "Colors",
    "prepare_camera_data_for_retrieval",
    "process_all_folders",
    "process_object_images",
]


def process_object_images(scan, walls, objects, wall_holes, *, include_walls=False):
    """Apply project crop settings and call the ported implementation."""
    enlarged_crop_objects = filter(None, os.environ.get("LR_ENLARGED_CROP_OBJECTS", "").split(","))
    return _process_object_images(
        scan,
        walls,
        objects,
        wall_holes,
        include_walls=include_walls,
        enlarged_crop_objects=enlarged_crop_objects,
    )
