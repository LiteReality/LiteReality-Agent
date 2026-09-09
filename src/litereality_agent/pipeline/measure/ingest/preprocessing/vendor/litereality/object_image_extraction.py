"""Object point cloud projection, ranking, and crop helpers from LiteReality."""

import glob
import json
import os
import pickle

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


class FrameStore:
    """One capture's per-frame data (intrinsic, pose, inverse pose, depth), loaded once.

    ``compute_mapping``, ``compute_mapping_for_3D_bbox`` and ``project_bbox_2d`` each take
    ``(scene_name, points, frame_id)`` and each did their own ``np.load`` / ``cv2.imread``. That
    signature is self-contained and easy to test, but it puts file I/O in the INNERMOST loop of a
    scan: ``crop_and_save_new`` sweeps every frame once PER OBJECT, so the same frames were read
    over and over. On a 28-object / 307-frame capture that is 25,788 ``np.load`` pairs and 17,192
    16-bit-PNG decodes of only 307 distinct images — measured at ~390 ms per object, of which just
    ~16 ms was the projection maths. Everything else was re-reading files already in memory.

    Frame data depends only on ``frame_id``, never on the object, so it belongs one level up. Hold
    a store across the object loop and each frame is read exactly once. The pose inverse is cached
    with it: ``np.linalg.inv`` was being recomputed three times per (object, frame) on 307
    matrices. Roughly 100 KB per frame, so ~30 MB for a 307-frame capture.

    Passing ``frames=None`` to any of those functions keeps the original read-per-call behaviour,
    so existing callers are unaffected.
    """

    def __init__(self, scene_name):
        self.scene_name = scene_name
        self._intrinsic = {}
        self._pose = {}
        self._pose_inv = {}
        self._depth = {}

    def intrinsic(self, frame_id):
        if frame_id not in self._intrinsic:
            self._intrinsic[frame_id] = np.load(
                f"input/rgbd/{self.scene_name}/intrinsic/intrinsic_{frame_id}.npy"
            )
        return self._intrinsic[frame_id]

    def pose(self, frame_id):
        if frame_id not in self._pose:
            self._pose[frame_id] = np.load(
                f"input/rgbd/{self.scene_name}/extrinsic/extrinsic_{frame_id}.npy"
            )
        return self._pose[frame_id]

    def pose_inv(self, frame_id):
        if frame_id not in self._pose_inv:
            self._pose_inv[frame_id] = np.linalg.inv(self.pose(frame_id))
        return self._pose_inv[frame_id]

    def depth(self, frame_id):
        """The depth map. Despite the ``.jpg`` name these are 16-bit greyscale PNGs, so this is a
        real PNG decode rather than the cheap JPEG read the extension suggests — which is why
        repeating it per object dominated the stage."""
        if frame_id not in self._depth:
            self._depth[frame_id] = cv2.imread(
                f"input/rgbd/{self.scene_name}/depth/frame_{frame_id}.jpg", cv2.IMREAD_UNCHANGED
            )
        return self._depth[frame_id]


def project_to_pixels(points, pose, fx, fy, cx, cy, bx=0.0, by=0.0, world_to_camera=None):
    """World points -> pixel coords, with the points that CANNOT be projected marked.

    ``world_to_camera`` is the already-inverted ``pose``; pass it to reuse a FrameStore's cached
    inverse instead of re-inverting the same matrix on every call.

    Returns ``(p, in_front)`` where ``p`` is the 4xN camera-space array with rows 0/1 replaced by
    pixel u/v, and ``in_front`` is the boolean mask of points that legitimately project.

    A pinhole projection divides by the camera-space depth ``p[2]``, and that is only defined for
    points IN FRONT of the camera. Two things go wrong without this guard:

      * ``p[2] == 0`` gives 0/0 -> nan, which is where the "invalid value encountered in divide"
        and "invalid value encountered in cast" warnings come from. Those are noisy but harmless:
        every comparison against nan is False, so the point falls out of the visibility mask.
      * ``p[2] < 0`` -- a point BEHIND the camera -- is the silent one. Dividing by a negative
        depth mirrors the point through the principal point and lands it at a perfectly plausible
        pixel, which then passes the in-bounds check as "visible". These functions decide which
        frames become an object's reference crops, so a phantom hit selects a frame the object is
        not actually in, and the reference image is built from it.
    """
    points_world = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
    if world_to_camera is None:
        world_to_camera = np.linalg.inv(pose)
    p = np.matmul(world_to_camera, points_world.T)  # [Xb, Yb, Zb, 1]: 4, n

    in_front = np.isfinite(p[2]) & (p[2] > 1e-6)
    u = np.full(p.shape[1], np.nan)
    v = np.full(p.shape[1], np.nan)
    np.divide((p[0] - bx) * fx, p[2], out=u, where=in_front)
    np.divide((p[1] - by) * fy, p[2], out=v, where=in_front)
    p[0], p[1] = u + cx, v + cy
    return p, in_front


def round_to_int(p):
    """Round pixel coords to int, without casting nan (which is undefined and warns)."""
    return np.round(np.where(np.isfinite(p), p, 0.0)).astype(int)


def compute_mapping_for_3D_bbox(scene_name, points, frame_id, frames=None):
    """
    :param points: N x 3 format
    :param depth: H x W format
    :param intrinsic: 3x3 format
    :param frames: optional FrameStore holding this capture's already-loaded frames
    :return: mapping, N x 3 format, (H,W,mask)
    """

    mapping = np.zeros((2, points.shape[0]), dtype=int)

    if frames is None:
        frames = FrameStore(scene_name)
    depth_intrinsic = frames.intrinsic(frame_id)
    depth = frames.depth(frame_id)
    pose = frames.pose(frame_id)

    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx, by = 0, 0

    p, in_front = project_to_pixels(points, pose, fx, fy, cx, cy, bx, by,
                                    world_to_camera=frames.pose_inv(frame_id))

    # out-of-image check
    mask = in_front * (p[0] > 0) * (p[1] > 0) \
                    * (p[0] < depth.shape[1]-1) \
                    * (p[1] < depth.shape[0]-1)
    pi = round_to_int(p)
    # directly keep the pixel whose depth!=0
    depth_mask = depth[pi[1][mask], pi[0][mask]] != 0
    mask[mask == True] = depth_mask
    
    # occlusion check:
    # trans_depth = depth[pi[1][mask], pi[0][mask]] / 1000
    # est_depth = p[2][mask]
    # occlusion_mask = np.abs(est_depth - trans_depth) <= 0.1
    # mask[mask == True] = occlusion_mask


    if mask.sum() == 0:
        return None, None, None


    mapping[0][mask] = p[1][mask]
    mapping[1][mask] = p[0][mask]
    
    # number of points that are visible
    num_visible = mask.sum()
    # get the bbox from mapping
    min_x = np.min(mapping[1][mask])
    max_x = np.max(mapping[1][mask])
    min_y = np.min(mapping[0][mask])
    max_y = np.max(mapping[0][mask])

    bbox = [min_x, min_y, max_x, max_y]

    return bbox, num_visible, mapping


def compute_mapping(scene_name, points, frame_id, frames=None):
    """
    :param points: N x 3 format
    :param depth: H x W format
    :param intrinsic: 3x3 format
    :param frames: optional FrameStore holding this capture's already-loaded frames
    :return: mapping, N x 3 format, (H,W,mask)
    """

    mapping = np.zeros((2, points.shape[0]), dtype=int)

    if frames is None:
        frames = FrameStore(scene_name)
    depth_intrinsic = frames.intrinsic(frame_id)
    depth = frames.depth(frame_id)
    pose = frames.pose(frame_id)

    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx, by = 0, 0

    p, in_front = project_to_pixels(points, pose, fx, fy, cx, cy, bx, by,
                                    world_to_camera=frames.pose_inv(frame_id))

    # out-of-image check
    mask = in_front * (p[0] > 0) * (p[1] > 0) \
                    * (p[0] < depth.shape[1]-1) \
                    * (p[1] < depth.shape[0]-1)

    pi = round_to_int(p)
    

    # directly keep the pixel whose depth!=0
    depth_mask = depth[pi[1][mask], pi[0][mask]] != 0
    mask[mask == True] = depth_mask
    
    # occlusion check:
    trans_depth = depth[pi[1][mask], pi[0][mask]] / 1000
    est_depth = p[2][mask]
    occlusion_mask = np.abs(est_depth - trans_depth) <= 0.1
    mask[mask == True] = occlusion_mask


    if mask.sum() == 0:
        return None, None, None


    mapping[0][mask] = p[1][mask]
    mapping[1][mask] = p[0][mask]
    
    # number of points that are visible
    num_visible = mask.sum()
    # get the bbox from mapping
    min_x = np.min(mapping[1][mask])
    max_x = np.max(mapping[1][mask])
    min_y = np.min(mapping[0][mask])
    max_y = np.max(mapping[0][mask])

    bbox = [min_x, min_y, max_x, max_y]

    return bbox, num_visible, mapping

def get_rotation_angle(points):

    top_rect_plot = points.copy()
    top_rect_plot.append(points[0])

    # Initialize list to store lines
    lines = []
    angles = []
    # Iterate over points and create lines
    for i in range(len(top_rect_plot) - 1):
        start = top_rect_plot[i]
        end = top_rect_plot[i + 1]
        if start[0] > end[0]:
            start, end = end, start
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        lines.append((start, end))
        angles.append(angle)
    return angles

def generate_eight_points(top_rect, y_values):

    """
    The bbox is represented by 8 points, the first 4 points are the top rectangle,
    """
    eight_points = np.zeros((8, 3))
    eight_points[:4, [0, 2]] = top_rect
    eight_points[:4, 1] = y_values[0]
    eight_points[4:, :] = eight_points[:4, :]
    eight_points[4:, 1] = y_values[1]
    return eight_points




def calculate_wall_corners(position, rotation, bbox):
    """Calculate 2D wall corners after applying rotation and translation."""
    # Convert rotations from degrees to radians
    rotation_rad = np.radians(rotation)
    
    # Define rotation matrices
    def rotation_matrix_x(angle):
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    
    def rotation_matrix_y(angle):
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    
    def rotation_matrix_z(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    # Combine rotation matrices
    rotation_matrix = rotation_matrix_x(rotation_rad[0]) @ rotation_matrix_y(rotation_rad[1]) @ rotation_matrix_z(rotation_rad[2])

    # Wall dimensions (width, depth) from bounding box
    width, _, depth = bbox

    corners = np.array([
        [-width / 2, 0, -depth / 2],
        [width / 2, 0, -depth / 2],
        [width / 2, 0, depth / 2],
        [-width / 2, 0, depth / 2]
    ])

    rotated_corners = (corners @ rotation_matrix.T) + position
    return rotated_corners[:, [0, 2]]  # Return X-Z coordinates for 2D plotting


def parse_objects(objects):
    angles_list = {}
    bboxes = {}
    for room_obj in objects:
        if "Floor" in room_obj["mesh_id"]: # skip the floor
            continue
        top_rect = room_obj["top_down_rect"]
        height = [-room_obj["bbox"][1]/2, room_obj["bbox"][1]/2] + room_obj["position"][1]
        angles_out = get_rotation_angle(top_rect)
        eight_points = generate_eight_points(top_rect, height)
        bboxes[room_obj["mesh_id"]] = eight_points
        angles_list[room_obj["mesh_id"]] = np.mean([angle for angle in angles_out if angle > 0])

    # print("obj------")
    # print("bboxes", bboxes)
    # print("average_positive_angle", average_positive_angle) 
    # print("obj--,,,--")
    return bboxes, angles_list


def parse_objects_wall(walls):
    angles_list = {}
    bboxes = {}

    objects = []
    for wall in walls:
        name = wall["file"].split("/")[-1].split(".")[0]
        object = wall["pose"]
        object["mesh_id"] = name 
        objects.append(object)

    for room_obj in objects:
        top_rect = room_obj["top_down_rect"]
        height = [-room_obj["bbox"][1]/2, room_obj["bbox"][1]/2] + room_obj["position"][1]
        angles_out = get_rotation_angle(top_rect)
        eight_points = generate_eight_points(top_rect, height)
        bboxes[room_obj["mesh_id"]] = eight_points
        angles_list[room_obj["mesh_id"]] = np.mean([angle for angle in angles_out if angle > 0])
    
    return bboxes, angles_list

def rotate_3d_points_along_y(points, angle_degrees):
    angle_radians = angle_degrees
    rotation_matrix = np.array([
        [np.cos(angle_radians), 0, np.sin(angle_radians)],
        [0, 1, 0],
        [-np.sin(angle_radians), 0, np.cos(angle_radians)]
    ])
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points

def filter_with_bbox(eight_points, pcd, factor = 1.0):
    max_x, max_y, max_z = np.max(eight_points, axis=0)
    min_x, min_y, min_z = np.min(eight_points, axis=0) 
    # scale the bbox to make sure the points are within the bbox

    x_center = (max_x + min_x) / 2
    y_center = (max_y + min_y) / 2
    z_center = (max_z + min_z) / 2

    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z

    max_x = x_center + x_range/2 * factor
    min_x = x_center - x_range/2 * factor
    max_y = y_center + y_range/2 * factor
    min_y = y_center - y_range/2 * factor
    max_z = z_center + z_range/2 * factor
    min_z = z_center - z_range/2 * factor

    mask = (pcd[:, 0] >= min_x) & (pcd[:, 0] <= max_x) & (pcd[:, 1] >= min_y) & (pcd[:, 1] <= max_y) & (pcd[:, 2] >= min_z) & (pcd[:, 2] <= max_z)
    return pcd[mask]

def crop_objs_with_bbox(bboxes, pcd_scene, rotation_y_list):

    """ to simplify the crop, we rotate the scene to align with x, z axis, 
    then crop the object with bbox, 
    and rotate back"""

    obj_pcd_list = {}
    for semantic, bbox in bboxes.items():
        rotation_y = rotation_y_list[semantic]
        pcd_scene_rotated = rotate_3d_points_along_y(pcd_scene, rotation_y)
        bbox_rotated = rotate_3d_points_along_y(bbox, rotation_y)
        obj_pcd = filter_with_bbox(bbox_rotated, pcd_scene_rotated)
        obj_pcd_rotated_back = rotate_3d_points_along_y(obj_pcd, -rotation_y)
        obj_pcd_list[semantic] = obj_pcd_rotated_back
    return obj_pcd_list

def get_pcd_objs(objects, pcd_scene):
    bbox, rotation_y = parse_objects(objects) 
    crop_out_objs = crop_objs_with_bbox(bbox, pcd_scene, rotation_y)
    return bbox, crop_out_objs

def get_pcd_objs_walls(objects, pcd_scene):

    bbox, rotation_y = parse_objects_wall(objects) 
    crop_out_objs = crop_objs_with_bbox(bbox, pcd_scene, rotation_y)
    return bbox, crop_out_objs, rotation_y

def object_view_quality(bbox, num_vis, max_vis, frame_w=256, frame_h=192):
    """★ Object-centric image-selection score — the LOCKED-IN method for picking which
    capture frames best show an object.

    Mirrors the harness `select_views.quality` (the 3-layer view-selection principle):
    rank a frame by how well the object FILLS the frame (size) and how CENTRED it is, gated
    by VISIBILITY — NOT by raw visible-point count (which favours close/dark/cluttered shots).

        score = size_term · (0.85 + 0.15·centredness) · vis           # in [0,1]

      size_term  = min(1, sqrt(bbox_area / frame_area) / 0.35)   # ~35% linear span = full detail
      centredness= 1 - dist(bbox centre, frame centre) / 70.71   # in % coords, 0..1
      vis        = min(1, (num_vis / best_frame_num_vis) / 0.5)  # ≥50% of the best frame -> full

    bbox is [x0,y0,x1,y1] in (frame_w, frame_h) pixels (default 256x192). Use it to rank a
    scan's frames per object and take the top few; the bbox itself is later DINO-refined by
    bbox_polish. Keep this consistent with select_views.quality if you tune it.
    """
    x0, y0, x1, y1 = bbox
    frac = max(0.0, x1 - x0) * max(0.0, y1 - y0) / float(frame_w * frame_h)
    size_term = min(1.0, (frac ** 0.5) / 0.35)
    cx = (x0 + x1) / 2.0 / frame_w * 100.0
    cy = (y0 + y1) / 2.0 / frame_h * 100.0
    centred = max(0.0, 1.0 - float(np.hypot(cx - 50.0, cy - 50.0)) / 70.71)
    vis = min(1.0, (num_vis / max(max_vis, 1)) / 0.5)
    return size_term * (0.85 + 0.15 * centred) * vis


def project_bbox_2d(scene_name, points, frame_id, W=256, H=192, frames=None):
    """2D extent of the 3D box corners projected into a frame — PURELY GEOMETRIC (no depth/occlusion
    masking, unlike compute_mapping_for_3D_bbox). This is what an ENLARGED crop needs: it spans the
    whole box even where the object has no depth returns (e.g. a base cabinet under a sink). Returns
    [x0,y0,x1,y1] clamped to the depth-image size, or None if the box isn't in front of the camera."""
    if frames is None:
        frames = FrameStore(scene_name)
    try:
        K = frames.intrinsic(frame_id)
        pose_inv = frames.pose_inv(frame_id)
    except Exception:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.concatenate([np.asarray(points, float), np.ones((len(points), 1))], axis=1)
    p = np.matmul(pose_inv, pts.T)   # 4 x N (camera space)
    front = p[2] > 1e-6
    if front.sum() < 2:
        return None
    px = (p[0][front] * fx) / p[2][front] + cx
    py = (p[1][front] * fy) / p[2][front] + cy
    x0, x1 = max(0, np.floor(px.min())), min(W, np.ceil(px.max()))
    y0, y1 = max(0, np.floor(py.min())), min(H, np.ceil(py.max()))
    if x1 <= x0 or y1 <= y0:
        return None
    return [int(x0), int(y0), int(x1), int(y1)]


def crop_and_save_new(scan_path, pcd_input, semantic, eight_points, top_k, frames=None):
    """Rank this object's frames. Pass `frames` — a FrameStore shared across the whole object loop
    — so the capture is read once for the scan rather than once per object; see FrameStore."""
    scene_name = scan_path.split("/")[-1]
    total_frame_numpy = len(glob.glob(f"{scan_path}/image/*.jpg"))
    if frames is None:
        frames = FrameStore(scene_name)
    W, H = 256, 192
    image_info = {}
    for i in range(total_frame_numpy):
        bbox, num_vis, mapping = compute_mapping(scene_name, pcd_input, i, frames)
        _, _, eight_point_mapping = compute_mapping_for_3D_bbox(scene_name, eight_points, i, frames)
        # bbox3d = geometric 2D extent of the FULL projected 3D box (depth-independent), so an
        # under-scanned unit (e.g. a sink whose base cabinet has few points) still spans the whole
        # object. The caller decides whether to use it for a given object.
        bbox3d = project_bbox_2d(scene_name, eight_points, i, W, H, frames)
        if bbox is not None and (bbox[2] > bbox[0]) and (bbox[3] > bbox[1]):
            bbox = [int(x) for x in bbox]
            image_info[i] = {"bbox": bbox, "num_vis": int(num_vis), "score": float(num_vis),
                             "mapping": mapping, "eight_point_mapping": eight_point_mapping,
                             "bbox3d": bbox3d}
    if not image_info:
        return []
    # rank by the object-centric view-selection score (size x centredness x visibility),
    # NOT raw visible-point count — see object_view_quality.
    max_vis = max(v["num_vis"] for v in image_info.values()) or 1
    for v in image_info.values():
        v["score"] = object_view_quality(v["bbox"], v["num_vis"], max_vis, W, H)
    top_k_images = sorted(image_info.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]
    return top_k_images

def _obj_has_faces(obj_path):
    """True if the OBJ declares any face — i.e. ASSIMP can actually build a mesh from it."""
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        return any(line.startswith("f ") for line in f)


def _obj_vertices(obj_path):
    """The `v` lines of an OBJ as an (N, 3) array — no faces needed."""
    vertices = []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    return np.asarray(vertices, dtype=float)


def get_pcd_from_obj(overall_scan_obj_path):
    """Scan vertices as (N, 3), whichever form scene_points.obj took.

    It is a textured mesh when the scan shipped textured_output.obj, and a vertex-only OBJ when
    the wrapper wrote it from pointcloud.pcd / depth frames. ASSIMP cannot build a mesh from the
    vertex-only form — it logs "all meshes are orphaned" and returns nothing — so check for faces
    first and only hand Open3D a file it can read.
    """
    if _obj_has_faces(overall_scan_obj_path):
        vertices = np.asarray(o3d.io.read_triangle_mesh(overall_scan_obj_path).vertices)
        if vertices.size:
            return vertices
    return _obj_vertices(overall_scan_obj_path)

if __name__ == "__main__":



    # Define scene and paths
    scene_name = "1_nov"

    scene_pcd_path = f"input/scan_geometry/{scene_name}/scene_points.obj"
    scene_data_path = f"input/scene_data/{scene_name}/objects.pkl"
    image_dir = f"input/rgbd/{scene_name}"
    parsed_image_dir = f"input/parsed_images_update/{scene_name}"

    # Load objects data and scene point cloud
    with open(scene_data_path, "rb") as f:
        objects = pickle.load(f)
    pcd_scene = get_pcd_from_obj(scene_pcd_path)

    # Select images for each object and crop objects with bounding box
    bbox, crop_out_objs = get_pcd_objs(objects, pcd_scene)
    
    # Process each object
    for object_id, pcd_input in crop_out_objs.items():

        top_k_images = crop_and_save_new(image_dir, pcd_input, object_id, bbox[object_id], top_k=10)

        # Define save paths for images and camera data
        object_save_dir = os.path.join(parsed_image_dir, object_id)
        camera_data_dir = os.path.join(object_save_dir, "camera")
        os.makedirs(object_save_dir, exist_ok=True)
        os.makedirs(camera_data_dir, exist_ok=True)

        for frame_id, image_info in top_k_images:
            # Load image and crop bounding box
            image_path = os.path.join(image_dir, "image", f"frame_{frame_id}.jpg")
            image = cv2.imread(image_path)
            
            bbox = image_info["bbox"]
            image_width, image_height = image.shape[1], image.shape[0]
            resized_bbox = [
                int(bbox[0] * image_width / 256), int(bbox[1] * image_height / 192),
                int(bbox[2] * image_width / 256), int(bbox[3] * image_height / 192)
            ]
            cropped_image = image[resized_bbox[1]:resized_bbox[3], resized_bbox[0]:resized_bbox[2]]

            # Load pose and intrinsic matrices
            pose_path = os.path.join(image_dir, "extrinsic", f"extrinsic_{frame_id}.npy")
            intrinsic_path = os.path.join(image_dir, "intrinsic", f"intrinsic_{frame_id}.npy")
            pose = np.load(pose_path)
            intrinsic = np.load(intrinsic_path)

            # Prepare frame metadata and save
            frame_info = {
                "semantic": object_id,
                "original_image_path": image_path,
                "pose": pose.tolist(),
                "intrinsic": intrinsic.tolist(),
                "bbox": bbox,
                "resized_bbox": resized_bbox
            }
            
            mapping = image_info["mapping"]
            # Create a copy to avoid overwriting
            dot_image = image.copy()

            for i in range(mapping.shape[1]):
                x, y = mapping[1][i], mapping[0][i]

                # resize the x, y to the original image size
                x = x * image_width / 256
                y = y * image_height / 192
                # Check if coordinates are within bounds
                height, width = dot_image.shape[:2]
                if 0 <= x < width and 0 <= y < height:
                    dot_image = cv2.circle(dot_image, (int(x), int(y)), 2, (0, 255, 0), -1)

            # Save the final image with all dots
            cv2.imwrite(os.path.join(object_save_dir, f"frame_{frame_id}_dot.jpg"), dot_image)
            # Save cropped image and metadata
            cv2.imwrite(os.path.join(object_save_dir, f"frame_{frame_id}.jpg"), cropped_image)
            with open(os.path.join(camera_data_dir, f"frame_{frame_id}.json"), "w") as f:
                json.dump(frame_info, f)

def canve_for_ref_images(base_folder):
    # generate stitched images for reference images
    
    # Iterate through each subfolder in the base folder
    for subfolder_name in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder_name)
        
        if os.path.isdir(subfolder_path):
            # Get the first 6 image files in the subfolder
            image_files = sorted([f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:6]
            if len(image_files) < 6:
                print(f"Not enough images in {subfolder_path}, skipping.")
                continue

            # Load and rotate each image
            images = [Image.open(os.path.join(subfolder_path, img)).rotate(-90, expand=True) for img in image_files]
            
            # Determine maximum width and height for consistent padding
            max_width = max(img.width for img in images)
            max_height = max(img.height for img in images)

            # Create blank images with padding for each
            padded_images = []
            for img in images:
                padded_img = Image.new("RGB", (max_width, max_height), (255, 255, 255))  # white background
                padded_img.paste(img, ((max_width - img.width) // 2, (max_height - img.height) // 2))
                padded_images.append(padded_img)

            # Stitch images in 2x3 grid
            grid_width = max_width * 3
            grid_height = max_height * 2
            grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))  # white background

            # Place images in grid
            for i, img in enumerate(padded_images):
                x = (i % 3) * max_width
                y = (i // 3) * max_height
                grid_image.paste(img, (x, y))

            # Save the stitched image
            output_path = os.path.join(subfolder_path, "stitched_image.jpg")
            grid_image.save(output_path)
            print(f"Saved stitched image in {output_path}")
        



def expand_window_bbox(window_points, factor=3.0):
    """
    Expand a window's 3D bounding box only along its thickness direction,
    scaling each point's thickness coordinate relative to the window center.
    
    Parameters:
      window_points: (8,3) numpy array of the eight corner points of the window.
                     Assumes the first 4 points lie on one face and the next 4 on the opposite face.
      factor: float, scaling factor for the thickness (e.g., 2.0 doubles the distance from the center along thickness).
      
    Returns:
      expanded_points: (8,3) numpy array of the expanded window bounding box.
    """
    window_points = np.array(window_points)
    # Compute the center of the window
    center = window_points.mean(axis=0)
    
    # Estimate the thickness direction using corresponding points on the two faces.
    thickness_vectors = []
    for i in range(4):
        thickness_vectors.append(window_points[i+4] - window_points[i])
    thickness_vectors = np.array(thickness_vectors)
    
    # Average the thickness vectors and normalize to get the unit thickness direction.
    avg_thickness_vec = thickness_vectors.mean(axis=0)
    thickness_dir = avg_thickness_vec / np.linalg.norm(avg_thickness_vec)
    
    expanded_points = []
    for pt in window_points:
        # Decompose the offset from the center into thickness and in-plane components.
        offset = pt - center
        d = np.dot(offset, thickness_dir)  # thickness component (can be positive or negative)
        in_plane = offset - d * thickness_dir  # remains unchanged
        
        # Scale the thickness component relative to the center.
        new_d = factor * d
        
        # Recombine to get the new point.
        new_pt = center + in_plane + new_d * thickness_dir
        expanded_points.append(new_pt)
    
    return np.array(expanded_points)

def get_wall_hole_points(wall_holes, rotation_y_list, bbox_walls, obj_points):
    # wall_holes and bbox_walls are dictionaries with wall names as keys.

    window_eight_points = {}

    # Loop over each wall that has a hole defined
    for key, hole in wall_holes.items():
        # Skip if no hole exists
        if not hole:
            continue

        # Get the hole's 2D dimension info (assumed to be a 2x2 array)
        # First row: [min_horizontal, min_vertical]
        # Second row: [max_horizontal, max_vertical]
        dim_all = np.array(hole["dimension"])
        num_hole = len(dim_all)

        for i in range(num_hole):
            type = hole["type"][i]
            dim = dim_all[i]
            min_coords = dim[0]  # lower left in the wall's 2D coordinate system
            max_coords = dim[1]  # upper right

            # Retrieve the 8 corner points of the wall bounding box
            pts = np.array(bbox_walls[key])
            # Compute the wall center (used as the origin for the wall's local coordinate system)
            wall_center = pts.mean(axis=0)

            # Define local axes from the wall's bounding box corners.
            # We assume the ordering: first four points for bottom face, next four for top face.
            p0 = pts[0]
            p1 = pts[1]
            p3 = pts[3]
            p4 = pts[4]  # corresponding top corner of p0

            # Two vectors along the bottom face.
            v1 = p1 - p0
            v2 = p3 - p0
            # The vertical direction (height)
            v3 = p4 - p0

            # Normalize vertical axis.
            vertical = v3 / np.linalg.norm(v3)

            # Decide which bottom edge represents the wall's thickness.
            # The shorter one is the thickness; the other will be our horizontal axis.
            if np.linalg.norm(v1) < np.linalg.norm(v2):
                thickness_length = np.linalg.norm(v1)
                thickness = v1 / thickness_length
                horizontal = v2 / np.linalg.norm(v2)
            else:
                thickness_length = np.linalg.norm(v2)
                thickness = v2 / thickness_length
                horizontal = v1 / np.linalg.norm(v1)

            # The hole's 2D coordinates are expressed in a system where:
            # - The origin is at the wall_center.
            # - The x-axis is the horizontal (longer) direction.
            # - The y-axis is vertical.
            # The four corners in 2D (order: bottom-left, bottom-right, top-right, top-left):
            corners_2d = np.array([
                [min_coords[0], min_coords[1]],
                [max_coords[0], min_coords[1]],
                [max_coords[0], max_coords[1]],
                [min_coords[0], max_coords[1]]
            ])

            # Compute the corresponding 3D points on the wall plane
            corners_3d_plane = np.array([
                wall_center + (corner[0] * horizontal + corner[1] * vertical)
                for corner in corners_2d
            ])

            # Extrude the 2D window along the wall's thickness.
            # We use half the wall thickness in each direction.
            extrude_distance = thickness_length / 2.0
            face1 = corners_3d_plane - extrude_distance * thickness
            face2 = corners_3d_plane + extrude_distance * thickness

            # Combine to get the 8 points for the window (first 4 from one face, next 4 from the other)
            window_pts = np.vstack((face1, face2))

            name = key + "_" + type + "_" + str(i)
            
            window_eight_points[name] = expand_window_bbox(window_pts) # expand the bbox to make sure the points are within the bbox


            # for key, points in window_eight_points.items():
            #     expanded_points = expand_window_bbox(points)
            #     window_eight_points[key] = expanded_points

    
    # load obj points as numpy array

    rotation_y_only_holes = {key: rotation_y_list[key.split("_")[0]] for key in window_eight_points.keys()}

    # crop the obj points with bbox

    obj_pcd_list = crop_objs_with_bbox(window_eight_points, obj_points, rotation_y_only_holes)
    return obj_pcd_list
