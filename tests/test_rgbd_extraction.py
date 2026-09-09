from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lrauthor.pipeline.measure.ingest.preprocessing.vendor.litereality import (
    roomplan as ingest_utils,
)


def _frame(image, depth):
    return SimpleNamespace(
        image_path_full=str(image),
        depth_path=str(depth),
        pose=np.eye(4),
        intrinsics=np.eye(3),
    )


def test_rgbd_copy_supports_paths_with_spaces(tmp_path, monkeypatch):
    source = tmp_path / "source files"
    source.mkdir()
    image = source / "source image.jpg"
    depth = source / "source depth.jpg"
    image.write_bytes(b"image")
    depth.write_bytes(b"depth")
    output = tmp_path / "output folder"
    monkeypatch.setattr(
        ingest_utils.scanner_capture,
        "load_scan_frames_no_video",
        lambda *_args, **_kwargs: [_frame(image, depth)],
    )

    ingest_utils.extract_rgbd("unused", str(output))

    assert (output / "image" / "frame_0.jpg").read_bytes() == b"image"
    assert (output / "depth" / "frame_0.jpg").read_bytes() == b"depth"


def test_rgbd_copy_reports_a_missing_source(tmp_path, monkeypatch):
    image = tmp_path / "missing image.jpg"
    depth = tmp_path / "missing depth.jpg"
    monkeypatch.setattr(
        ingest_utils.scanner_capture,
        "load_scan_frames_no_video",
        lambda *_args, **_kwargs: [_frame(image, depth)],
    )

    with pytest.raises(FileNotFoundError):
        ingest_utils.extract_rgbd("unused", str(tmp_path / "output"))
