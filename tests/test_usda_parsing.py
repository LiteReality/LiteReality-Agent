from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lrauthor.pipeline.measure.ingest.preprocessing.vendor.litereality.roomplan import (
    get_object_pose,
)

IDENTITY = "[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]"


def test_object_pose_parses_numeric_usda_literals(tmp_path: Path):
    usda = tmp_path / "object.usda"
    usda.write_text(
        f"point3f[] points = [[0, 0, 0], [1, 2, 3]]\nmatrix4d xformOp:transform = {IDENTITY}\n",
        encoding="utf-8",
    )

    pose = get_object_pose(str(usda))

    np.testing.assert_array_equal(pose["bbox"], [1, 2, 3])
    np.testing.assert_array_equal(pose["rotation"], [0, 0, 0])


def test_object_pose_does_not_execute_usda_values(tmp_path: Path):
    marker = tmp_path / "executed"
    usda = tmp_path / "object.usda"
    usda.write_text(
        f'point3f[] points = (__import__("pathlib").Path({str(marker)!r})'
        '.write_text("yes"), [[0, 0, 0], [1, 1, 1]])[1]\n'
        f"matrix4d xformOp:transform = {IDENTITY}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid USDA literal"):
        get_object_pose(str(usda))

    assert not marker.exists()
