"""ARKit capture cameras -> Blender cameras.  Run INSIDE Blender (bpy).

    import sys; sys.path.insert(0, "evidence/helpers"); import arkit_cameras
    cams = arkit_cameras.add_cameras("evidence/scan")          # one Blender camera per evidence/scan/frame_NNNNN.json
    arkit_cameras.render(cams, "out/renders", frames=[4, 12, 25, 36], res_div=2)

Conventions (verified against this dataset):
  * evidence/scan/frame_NNNNN.json["cameraPoseARFrame"] is a ROW-MAJOR 4x4 camera->world matrix in ARKit
    world coordinates (Y up, metres). Translation is the last column.
  * pointcloud.pcd and the RoomPlan usdz share that same ARKit world frame.
  * Blender is Z-up: world_blender = Rx(+90deg) @ world_arkit.   (y_arkit -> z_blender, z_arkit -> -y_blender)
  * ARKit camera looks down its local -Z with +Y up, which is exactly Blender's camera convention,
    so the Blender camera matrix is simply  Rx(90) @ pose.
  * "intrinsics" is row-major 3x3 [fx 0 cx; 0 fy cy; 0 0 1] for the LANDSCAPE 1920x1440 jpg.
    (The phone was held portrait, so the jpgs look rotated; cx~960, cy~720 confirms landscape storage.)
If you build the scene in the ARKit frame instead, apply the same Rx(90) to your geometry, or set
C = Matrix.Identity(4) below — just be consistent, the renders must line up with the photographs.
"""
import json, math, os, re
import bpy
from mathutils import Matrix

SENSOR_W = 36.0
C = Matrix.Rotation(math.radians(90), 4, "X")   # ARKit (Y-up) -> Blender (Z-up)


def pose_matrix(flat16):
    return Matrix((flat16[0:4], flat16[4:8], flat16[8:12], flat16[12:16]))


def image_size(scan_dir):
    for f in sorted(os.listdir(scan_dir)):
        if re.match(r"frame_\d+\.jpg$", f):
            img = bpy.data.images.load(os.path.join(scan_dir, f))
            w, h = img.size
            bpy.data.images.remove(img)
            return w, h
    return 1920, 1440


def add_cameras(scan_dir="evidence/scan", collection="ARKitCameras"):
    """Create cam_NNNNN for every frame json. Returns {frame_index: camera_object}."""
    w, h = image_size(scan_dir)
    coll = bpy.data.collections.get(collection) or bpy.data.collections.new(collection)
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    cams = {}
    for f in sorted(os.listdir(scan_dir)):
        if not re.match(r"frame_\d+\.json$", f):
            continue
        d = json.load(open(os.path.join(scan_dir, f)))
        idx = d.get("frame_index", int(re.search(r"\d+", f).group()))
        fx, _, cx, _, fy, cy, _, _, _ = d["intrinsics"]
        cd = bpy.data.cameras.new(f"cam_{idx:05d}")
        cd.sensor_fit = "HORIZONTAL"
        cd.sensor_width = SENSOR_W
        cd.lens = fx * SENSOR_W / w
        cd.shift_x = (w * 0.5 - cx) / w
        cd.shift_y = (cy - h * 0.5) / w
        cd.clip_start, cd.clip_end = 0.01, 100.0
        ob = bpy.data.objects.new(cd.name, cd)
        ob.matrix_world = C @ pose_matrix(d["cameraPoseARFrame"])
        coll.objects.link(ob)
        cams[idx] = ob
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = w, h
    return cams


def render(cams, out_dir="out/renders", frames=(4, 12, 25, 36), res_div=2):
    """Render the current scene from the given capture frames (landscape, same size as the jpgs / res_div)."""
    os.makedirs(out_dir, exist_ok=True)
    sc = bpy.context.scene
    sc.render.resolution_percentage = max(1, int(100 / res_div))
    for fi in frames:
        sc.camera = cams[fi]
        sc.render.filepath = os.path.join(out_dir, f"frame_{fi:05d}.png")
        bpy.ops.render.render(write_still=True)
        print("rendered", sc.render.filepath)
    sc.render.resolution_percentage = 100
