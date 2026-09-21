import json

import trimesh

from litereality_agent.pipeline.room_qc import support_audit as A
from litereality_agent.pipeline.room_qc.support_audit import support_graph, vertical_contacts


def test_strict_openings_require_real_wall_support_not_an_exemption():
    objects = [
        {"id": "Wall2", "category": "wall"},
        {"id": "Door0", "category": "door", "source_glb": "Wall2_Door_0.glb"},
        {"id": "Window1", "category": "window"},
    ]
    edges, findings = support_graph(objects, strict=True)
    assert edges["Door0"] == {"target": "Wall2", "relation": "attached_to", "inferred": True}
    assert {"id": "Window1", "kind": "undeclared_support"} in findings


def test_strict_vertical_contact_rejects_one_centimeter_floating():
    floor = box((2, 2, 0.1), (0, 0, -0.05))
    child = box((0.5, 0.5, 0.5), (0, 0, 0.26))
    assert vertical_contacts(child, floor)["contact_samples"] > 0
    assert (
        vertical_contacts(child, floor, tolerance=0.005, sink_tolerance=0.005)["contact_samples"]
        == 0
    )


def box(extents, center):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def test_floating_member_inside_merged_mesh_cannot_hide_behind_grounded_member():
    floor = box((4, 4, 0.1), (0, 0, -0.05))
    group = trimesh.util.concatenate(
        [box((0.5, 0.5, 1), (-1, 0, 0.5)), box((0.5, 0.5, 1), (1, 0, 0.8))]
    )
    findings = A.disconnected_findings("MergedChairs", group, floor)
    assert len(findings) == 1 and findings[0]["kind"] == "unsupported_disconnected_member"


def test_touching_stack_has_a_contact_path():
    floor = box((4, 4, 0.1), (0, 0, -0.05))
    group = trimesh.util.concatenate(
        [box((0.5, 0.5, 0.5), (0, 0, 0.25)), box((0.5, 0.5, 0.5), (0, 0, 0.75))]
    )
    assert A.disconnected_findings("Stack", group, floor) == []


def test_attachment_to_missing_wall_is_not_accepted():
    _, findings = support_graph([{"id": "Picture", "attached_to": "MissingWall"}])
    assert any(f["kind"] == "unknown_support" for f in findings)


def test_cycle_is_not_a_support_system():
    _, findings = support_graph([{"id": "A", "rests_on": "B"}, {"id": "B", "rests_on": "A"}])
    assert any(f["kind"] == "support_cycle" for f in findings)


def test_rooted_support_chain_and_inferred_floor():
    edges, findings = support_graph(
        [
            {"id": "Floor", "category": "floor"},
            {"id": "Table", "category": "table"},
            {"id": "Mat", "rests_on": "Table"},
            {"id": "Cup", "rests_on": "Mat"},
        ]
    )
    assert not findings
    assert edges["Table"]["inferred"]


def test_vertical_contacts_distinguish_touch_float_and_off_edge():
    table = box((1, 1, 0.1), (0, 0, 0.75))
    cup = box((0.1, 0.1, 0.2), (0, 0, 0.9))
    assert vertical_contacts(cup, table)["contact_samples"] > 0
    cup.apply_translation((0, 0, 0.05))
    assert vertical_contacts(cup, table)["contact_samples"] == 0
    cup.apply_translation((2, 0, -0.05))
    assert vertical_contacts(cup, table)["contact_samples"] == 0


def test_rays_use_mesh_not_support_bbox_over_empty_space():
    # Bounding box spans the gap, but there is no surface underneath the cup.
    sides = trimesh.util.concatenate(
        [box((0.1, 1, 0.1), (-0.45, 0, 0.75)), box((0.1, 1, 0.1), (0.45, 0, 0.75))]
    )
    cup = box((0.1, 0.1, 0.2), (0, 0, 0.9))
    assert vertical_contacts(cup, sides)["contact_samples"] == 0


def test_inside_a_solid_support_cannot_rest_on_its_bottom_face():
    support = box((1, 1, 1), (0, 0, 0.5))
    buried = box((0.1, 0.1, 0.1), (0, 0, 0.055))
    assert vertical_contacts(buried, support)["contact_samples"] == 0


def test_draped_textile_is_supported_by_top_patch_not_hanging_hem():
    support = box((1, 1, 0.1), (0, 0, 0.75))
    textile = trimesh.util.concatenate(
        [box((1.04, 0.6, 0.004), (0, 0, 0.802)), box((0.004, 0.6, 0.3), (0.518, 0, 0.65))]
    )
    assert not vertical_contacts(textile, support)["contact_samples"]
    check = vertical_contacts(textile, support, flexible=True)
    assert check["contact_samples"] >= 3
    assert check["contact_hull_area_m2"] > 0.01
    textile.apply_translation((0, 0, 0.1))
    assert not vertical_contacts(textile, support, flexible=True)["contact_samples"]


def test_flexible_sampling_keeps_bottom_when_child_normals_are_reversed():
    table = box((1, 1, 0.1), (0, 0, 0.75))
    pillow = box((0.3, 0.3, 0.2), (0, 0, 0.9))
    pillow.invert()
    assert vertical_contacts(pillow, table, flexible=True)["contact_samples"] >= 4


def test_attachment_requires_actual_mesh_contact(tmp_path, monkeypatch):
    objects = [
        {"id": "Wall", "category": "wall"},
        {"id": "Picture", "category": "picture", "attached_to": "Wall"},
    ]
    (tmp_path / "room_layout.json").write_text(json.dumps({"objects": objects}))
    meshes = {"Wall": box((0.1, 2, 2), (0, 0, 1)), "Picture": box((0.04, 0.5, 0.5), (0.2, 0, 1))}
    monkeypatch.setattr(A, "mesh_bodies", lambda *args: meshes)
    report = A.audit(tmp_path)
    assert not report["pass"]
    assert report["support_findings"][0]["kind"] == "attachment_gap"
    meshes["Picture"].apply_translation((-0.13, 0, 0))
    assert A.audit(tmp_path)["pass"]


def test_only_measured_separation_can_contradict_wall_plane_report(tmp_path, monkeypatch):
    objects = [{"id": "Wall", "category": "wall"}, {"id": "OtherWall", "category": "wall"}]
    (tmp_path / "room_layout.json").write_text(json.dumps({"objects": objects}))
    meshes = {"Wall": box((0.1, 2, 2), (0, 0, 1)), "OtherWall": box((0.1, 2, 2), (1, 0, 1))}
    monkeypatch.setattr(A, "mesh_bodies", lambda *args: meshes)
    legacy = tmp_path / "legacy.json"
    finding = {"id": "OtherWall", "wall": "Wall", "kind": "wall_clash"}
    legacy.write_text(json.dumps({"ok": True, "failing": [finding]}))
    report = A.audit(tmp_path, legacy)
    assert report["pass"]
    assert len(report["legacy_contradicted_by_finite_geometry"]) == 1
    meshes["OtherWall"].apply_translation((-1, 0, 0))
    report = A.audit(tmp_path, legacy)
    assert not report["pass"]
    assert report["legacy_unresolved"] == [finding]


def test_contact_surface_supersedes_headboard_height_but_not_real_gap(tmp_path, monkeypatch):
    objects = [
        {"id": "Floor", "category": "floor"},
        {"id": "Bed", "category": "bed", "rests_on": "Floor"},
        {"id": "Pillow", "category": "pillow", "rests_on": "Bed"},
    ]
    (tmp_path / "room_layout.json").write_text(json.dumps({"objects": objects}))
    meshes = {
        "Floor": box((3, 3, 0.1), (0, 0, -0.05)),
        "Bed": trimesh.util.concatenate(
            [box((1, 1, 0.8), (0, 0, 0.4)), box((1, 0.1, 1.2), (0, 0.55, 0.6))]
        ),
        "Pillow": box((0.3, 0.2, 0.1), (0, 0.2, 0.85)),
    }
    monkeypatch.setattr(A, "mesh_bodies", lambda *args: meshes)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"ok": True, "failing": [{"id": "Pillow", "kind": "sunk"}]}))
    report = A.audit(tmp_path, legacy)
    assert report["pass"]
    assert len(report["legacy_support_findings_rechecked_on_contact_surface"]) == 1
    meshes["Pillow"].apply_translation((0, 0, 0.05))
    report = A.audit(tmp_path, legacy)
    assert not report["pass"]
    assert any(f["id"] == "Pillow" for f in report["support_findings"])
