"""Crop-stage reuse and the stitching that feeds it.

Both behaviours here are things a re-run gets wrong silently, so they are worth pinning: the
stage used to re-crop unconditionally, and the stitcher used to consume its own output.
"""

from __future__ import annotations

import json
import pickle

import pytest
from PIL import Image

from lrauthor.pipeline.measure import paths as config
from lrauthor.pipeline.measure.ingest.crop import crop_objects
from lrauthor.pipeline.measure.ingest.preprocessing.vendor.litereality import (
    scene_preprocessing,
)

SCAN = "office"


@pytest.fixture
def work_root(tmp_path, monkeypatch):
    """A minimal work tree: scene_data pkls, one rgbd frame, and an empty crop dir."""
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path))
    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path))
    monkeypatch.delenv("LR_ENLARGED_CROP_OBJECTS", raising=False)
    config.enter_work_root(SCAN)

    scene = config.scene_data_dir(SCAN)
    scene.mkdir(parents=True, exist_ok=True)
    for name in ("walls", "objects", "wall_holes", "floor"):
        with (scene / f"{name}.pkl").open("wb") as handle:
            pickle.dump([{"mesh_id": "Table0"}], handle)

    frames = config.input_root() / "rgbd" / SCAN / "image"
    frames.mkdir(parents=True, exist_ok=True)
    (frames / "frame_0.jpg").write_bytes(b"x")

    parsed = config.parsed_images_dir(SCAN)
    (parsed / "Table0").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _stamp(include_walls: bool = False) -> None:
    """Write the stamp `crop()` would leave behind, without running the real crop."""
    parsed = config.parsed_images_dir(SCAN)
    (parsed / crop_objects.STAMP).write_text(
        json.dumps(crop_objects._input_fingerprint(SCAN, include_walls)), encoding="utf-8"
    )


def test_crops_not_reused_without_a_stamp(work_root):
    """Crops from before this stamp existed must be rebuilt, not trusted."""
    assert crop_objects.crops_current(SCAN) is False


def test_crops_reused_when_inputs_are_unchanged(work_root):
    _stamp()
    assert crop_objects.crops_current(SCAN) is True


def test_box_merge_rewriting_objects_invalidates_crops(work_root):
    """box_merge runs between extract and crop and rewrites objects.pkl, so crops built against
    the pre-merge object set must not be reused — an existence check alone would reuse them."""
    _stamp()
    scene = config.scene_data_dir(SCAN)
    with (scene / "objects.pkl").open("wb") as handle:
        pickle.dump([{"mesh_id": "Table0"}, {"mesh_id": "StorageRun0"}], handle)
    assert crop_objects.crops_current(SCAN) is False


def test_enlarged_crop_objects_invalidates_crops(work_root, monkeypatch):
    """box_merge also sets $LR_ENLARGED_CROP_OBJECTS, which changes the crop rectangle."""
    _stamp()
    monkeypatch.setenv("LR_ENLARGED_CROP_OBJECTS", "StorageRun0")
    assert crop_objects.crops_current(SCAN) is False


def test_include_walls_invalidates_crops(work_root):
    _stamp(include_walls=False)
    assert crop_objects.crops_current(SCAN, include_walls=True) is False


def test_new_frames_invalidate_crops(work_root):
    _stamp()
    (config.input_root() / "rgbd" / SCAN / "image" / "frame_1.jpg").write_bytes(b"x")
    assert crop_objects.crops_current(SCAN) is False


def test_empty_crop_dir_is_not_reusable(work_root):
    """A stamp with no object folders beside it means an interrupted run, not a finished one."""
    _stamp()
    (config.parsed_images_dir(SCAN) / "Table0").rmdir()
    assert crop_objects.crops_current(SCAN) is False


@pytest.mark.parametrize("n_crops", [1, 2, 3, 4, 5])
def test_stitching_is_idempotent(tmp_path, n_crops):
    """Re-stitching a folder must not consume the previous stitched_image.jpg.

    `extract_ranking` sorts it last, so with 4+ crops it fell outside the top-4 selection and the
    feedback never showed. With fewer — the wall doors/windows, which often rank 1-3 frames — it
    was pulled back in and every re-crop stitched the stitch, growing without bound until PIL
    refused to open it and the object lost its evidence sheet entirely.
    """
    folder = tmp_path / "Wall2_Door_0"
    folder.mkdir()
    for i in range(n_crops):
        Image.new("RGB", (40, 30), (i * 10, 0, 0)).save(folder / f"frame_{i}_ranking_{i}.jpg")

    scene_preprocessing.stitch_top_images(str(folder))
    first = Image.open(folder / scene_preprocessing.STITCHED_NAME).size

    for _ in range(3):
        scene_preprocessing.stitch_top_images(str(folder))
        assert Image.open(folder / scene_preprocessing.STITCHED_NAME).size == first
