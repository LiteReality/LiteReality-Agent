"""`--skip-references` must reuse the reference manifests, not silently empty them.

The reconstruct stage re-enters scene_init.flow only to reach classify/build, and used to
re-derive all three reference sets over crops that `--skip-crop` had already frozen. Chair grouping
is an LLM judgment, so it comes back different often enough to matter — four runs of the same five
chairs gave 2/3/3/2 clusters — and `cluster_and_generate` rewrites chair_clusters.json
unconditionally, so the later stage could overwrite the grouping ingest committed to.

The failure mode of the fix is worse than the bug: if the reuse path returned empty results the
run would proceed with ZERO objects and look like a clean no-op. So both directions are pinned —
reuse loads what is on disk, and a missing manifest falls back to regenerating.
"""

from __future__ import annotations

import json

import pytest

from litereality_agent.pipeline.measure import flow

CLUSTERS = {
    "chair_count": 5,
    "clusters": [
        {"cluster_id": "ChairCluster0", "members": ["Chair0", "Chair2"], "representative": "Chair0"},
        {"cluster_id": "ChairCluster1", "members": ["Chair1"], "representative": "Chair1"},
    ],
}
OBJECTS = {"scan": "Room", "objects": [{"object": "Table0"}, {"object": "Table1"}]}
OPENINGS = {"scan": "Room", "openings": [{"opening": "Wall1_Door_0"}]}


def _write(tmp_path, *, openings=True):
    """Lay out the three manifests the way a completed ingest leaves them."""
    obj = tmp_path / "object_refs" / "Room"
    chair = tmp_path / "chair_clusters" / "Room"
    op = tmp_path / "opening_refs" / "Room"
    for d in (obj, chair, op):
        d.mkdir(parents=True)
    (obj / "object_references.json").write_text(json.dumps(OBJECTS))
    (chair / "chair_clusters.json").write_text(json.dumps(CLUSTERS))
    if openings:
        (op / "opening_references.json").write_text(json.dumps(OPENINGS))
    return obj, chair, op


def _point_config_at(monkeypatch, tmp_path):
    monkeypatch.setattr(flow.config, "object_refs_root", lambda: tmp_path / "object_refs")
    monkeypatch.setattr(flow.config, "chair_clusters_root", lambda: tmp_path / "chair_clusters")
    monkeypatch.setattr(flow.config, "opening_refs_root", lambda: tmp_path / "opening_refs")


def test_reuses_manifests_on_disk(monkeypatch, tmp_path):
    _write(tmp_path)
    _point_config_at(monkeypatch, tmp_path)

    assert flow._references_on_disk("Room") is True

    objects = flow._load_reference_json(
        tmp_path / "object_refs" / "Room" / "object_references.json", {"scan": "Room", "objects": []}
    )
    chairs = flow._load_reference_json(
        tmp_path / "chair_clusters" / "Room" / "chair_clusters.json", {"chair_count": 0, "clusters": []}
    )

    # The whole point: the reused grouping is the one ingest committed to, not a fresh judgment.
    assert [o["object"] for o in objects["objects"]] == ["Table0", "Table1"]
    assert chairs["chair_count"] == 5
    assert [c["cluster_id"] for c in chairs["clusters"]] == ["ChairCluster0", "ChairCluster1"]


def test_missing_manifest_falls_back_to_regenerating(monkeypatch, tmp_path):
    """A partial tree must NOT be reused — routing and reconstruction would disagree."""
    _write(tmp_path)
    _point_config_at(monkeypatch, tmp_path)
    (tmp_path / "chair_clusters" / "Room" / "chair_clusters.json").unlink()

    assert flow._references_on_disk("Room") is False


def test_room_without_openings_reuses_cleanly(monkeypatch, tmp_path):
    """No doors or windows is legitimate — an absent openings manifest is not an error."""
    _write(tmp_path, openings=False)
    _point_config_at(monkeypatch, tmp_path)

    assert flow._references_on_disk("Room") is True
    assert flow._load_reference_json(
        tmp_path / "opening_refs" / "Room" / "opening_references.json", {"openings": []}
    ) == {"openings": []}


def test_corrupt_manifest_raises_rather_than_emptying_the_run(monkeypatch, tmp_path):
    """Swallowing a parse error would drop every object and read as a clean no-op."""
    _write(tmp_path)
    _point_config_at(monkeypatch, tmp_path)
    (tmp_path / "object_refs" / "Room" / "object_references.json").write_text("{not json")

    with pytest.raises(ValueError):
        flow._load_reference_json(
            tmp_path / "object_refs" / "Room" / "object_references.json",
            {"scan": "Room", "objects": []},
        )
