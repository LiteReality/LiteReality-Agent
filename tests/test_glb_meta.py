"""glb_meta — the collision model's facts carried inside the glb as glTF `extras`.

The point of this module is that `check_glb(path)` is a function of ONE file. These tests pin the
round-trip and, more importantly, the two failure modes that would make it silently wrong: a glb
with no metadata must fall BACK to the caller's SHELL rather than report a clean room, and a
rewritten glb must still be a valid binary (chunk padding is easy to get wrong and every loader
rejects the result).
"""

from __future__ import annotations

import json
import struct

import pytest
import trimesh

from litereality_agent.agent.tools.check_collisions.source import collision_mesh as sc
from litereality_agent.room_ops import glb_meta

_SHELL = {
    "floor_z": -1.6879,
    "ceiling_z": 0.9821,
    "walls": {"Wall0": {"start": [-4.2259, -3.4376], "end": [-5.3818, -0.9763],
                        "thickness": 0.0001}},
    "openings": {"Door0": {"wall": "Wall0", "offset": 0.5, "width": 0.9, "height": 2.0,
                           "sill": 0.0}},
    "objects": {"Chair0": {"category": "chair", "center": [0, 0, 0], "size": [0.5, 0.5, 0.8],
                           "yaw": 25.16},
                "Television0": {"category": "television", "center": [1, 0, 1],
                                "size": [1.4, 0.1, 0.8], "yaw": -154.84}},
}


def _make_glb(tmp_path, names=("Wall0", "Chair0", "Television0", "Door0")):
    scene = trimesh.Scene()
    for i, n in enumerate(names):
        m = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
        m.apply_translation((i * 2.0, 0, 0))
        scene.add_geometry(m, node_name=n, geom_name=f"{n}_mesh")
    p = tmp_path / "Room.glb"
    p.write_bytes(scene.export(file_type="glb"))
    return p


def test_round_trip_recovers_every_field(tmp_path):
    glb = _make_glb(tmp_path)
    counts = glb_meta.stamp(glb, _SHELL)
    assert counts == {"walls": 1, "objects": 2, "openings": 1}

    back = glb_meta.shell_from_glb(glb)
    assert back["floor_z"] == pytest.approx(-1.6879)
    assert back["ceiling_z"] == pytest.approx(0.9821)
    assert back["walls"]["Wall0"]["start"] == pytest.approx([-4.2259, -3.4376])
    assert back["walls"]["Wall0"]["end"] == pytest.approx([-5.3818, -0.9763])
    # the category that node-name matching CANNOT recover — Television matches no furniture pattern
    assert back["objects"]["Television0"]["category"] == "television"
    assert back["objects"]["Chair0"]["yaw"] == pytest.approx(25.16)
    assert back["openings"]["Door0"]["width"] == pytest.approx(0.9)


def test_unstamped_glb_reports_nothing_rather_than_guessing(tmp_path):
    """An older glb must return {} so the caller falls back to Room.py — never a false clean room."""
    assert glb_meta.shell_from_glb(_make_glb(tmp_path)) == {}


def test_explicit_shell_wins_over_the_embedded_one(tmp_path):
    """`shell_for` must prefer the caller's SHELL: Room.py is the source of truth, the glb a copy."""
    glb = _make_glb(tmp_path)
    glb_meta.stamp(glb, _SHELL)
    explicit = {"floor_z": 99.0, "walls": {}, "objects": {}}
    assert sc.shell_for(glb, explicit) is explicit
    assert sc.shell_for(glb)["floor_z"] == pytest.approx(-1.6879)


def test_stamped_glb_is_still_a_valid_binary(tmp_path):
    """Chunk padding: JSON pads with spaces, BIN with zeros, header length must match."""
    glb = _make_glb(tmp_path)
    before = trimesh.load(glb, process=False)
    glb_meta.stamp(glb, _SHELL)

    raw = glb.read_bytes()
    assert raw[:4] == b"glTF"
    assert struct.unpack("<I", raw[8:12])[0] == len(raw), "header length must match the file"
    assert len(raw) % 4 == 0, "chunks must stay 4-byte aligned"

    after = trimesh.load(glb, process=False)
    assert len(after.geometry) == len(before.geometry), "geometry must survive the rewrite"


def test_stamp_is_idempotent(tmp_path):
    glb = _make_glb(tmp_path)
    glb_meta.stamp(glb, _SHELL)
    first = glb.read_bytes()
    glb_meta.stamp(glb, _SHELL)
    assert glb.read_bytes() == first, "re-stamping must not grow or churn the file"


def test_schema_mismatch_is_treated_as_absent(tmp_path):
    """A future/foreign schema must fall back, not be half-read into a wrong collision model."""
    glb = _make_glb(tmp_path)
    glb_meta.stamp(glb, _SHELL)
    gltf, binary = glb_meta.read_chunks(glb)
    gltf["scenes"][0]["extras"]["lr_schema"] = "litereality/collision/999"
    glb_meta.write_chunks(glb, gltf, binary)
    assert glb_meta.shell_from_glb(glb) == {}


def test_binary_chunk_is_preserved(tmp_path):
    """The mesh data must come through byte-identical — a stamp is metadata only."""
    glb = _make_glb(tmp_path)
    _, bin_before = glb_meta.read_chunks(glb)
    glb_meta.stamp(glb, _SHELL)
    _, bin_after = glb_meta.read_chunks(glb)
    assert bin_after[:len(bin_before)] == bin_before


def test_extras_land_where_a_gltf_reader_expects_them(tmp_path):
    """Written as standard glTF `extras`, so Blender/three.js/any loader can read them too."""
    glb = _make_glb(tmp_path)
    glb_meta.stamp(glb, _SHELL)
    b = glb.read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + n])
    by_name = {nd.get("name"): nd.get("extras") for nd in j["nodes"] if nd.get("name")}
    assert by_name["Wall0"]["lr_kind"] == "wall"
    assert by_name["Chair0"]["lr_category"] == "chair"
    assert j["scenes"][j.get("scene", 0)]["extras"]["lr_schema"] == glb_meta.SCHEMA
