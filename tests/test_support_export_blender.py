"""Real Blender regression: authored contents follow the lift, including after GLB export."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from litereality_agent.pipeline.room_qc.export_support import check, glb_document


@pytest.mark.blender
def test_named_support_part_survives_blender_export(tmp_path):
    from litereality_agent.room_ops.paths import find_blender

    source = Path(__file__).resolve().parents[1] / "src"
    script = tmp_path / "scene.py"
    script.write_text(
        """
import sys, json
sys.path.insert(0, SOURCE)
import bpy
from litereality_agent.room_ops.compile.build_room import RoomScene, group_fixture
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
table=bpy.data.objects.new('Table',None)
table['room_id']='Table'; table['category']='table'
bpy.context.scene.collection.objects.link(table)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,.75))
top=bpy.context.object; top.name='Desktop'; top.dimensions=(1,1,.1); top.parent=table
top.keyframe_insert(data_path='location',frame=0)
top.location.z=1.1; top.keyframe_insert(data_path='location',frame=10)
bpy.context.scene.frame_set(0)
bpy.ops.mesh.primitive_cube_add(size=.1, location=(0,0,.85))
cup=group_fixture('Cup','mug',[bpy.context.object],rests_on='Table',support_part='Desktop')
bpy.context.view_layer.update()
before=cup.matrix_world.copy()
scene=object.__new__(RoomScene)
scene.out_glb=OUT
scene.bind_supports()
assert max(abs(cup.matrix_world[i][j]-before[i][j]) for i in range(4) for j in range(4))<1e-6
relative=top.matrix_world.inverted() @ cup.matrix_world
for frame in (0,2,5,8,10,0):
    bpy.context.scene.frame_set(frame)
    now=top.matrix_world.inverted() @ cup.matrix_world
    assert max(abs(now[i][j]-relative[i][j]) for i in range(4) for j in range(4))<1e-6
scene.export_glb()
assert scene.objects['Cup']['support_part']=='Desktop'
assert scene.objects['Table']['bbox_max'][2]<.81
open(LAYOUT,'w').write(json.dumps(list(scene.objects.values())))
""".replace("LAYOUT", repr(str(tmp_path / "objects.json")))
        .replace("SOURCE", repr(str(source)))
        .replace("OUT", repr(str(tmp_path / "Room.glb")))
    )
    result = subprocess.run(
        [find_blender(), "-b", "--python-exit-code", "1", "--python", str(script)],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    doc = glb_document(tmp_path / "Room.glb")
    objects = json.loads((tmp_path / "objects.json").read_text())
    assert check(doc, objects, {"assets": [
        {"object": "Table", "kind": "articulated", "glb": "Room.glb"}]}) == []
