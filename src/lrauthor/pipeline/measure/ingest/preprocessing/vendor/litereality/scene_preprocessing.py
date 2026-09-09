# -*- coding: utf-8 -*-
"""
LiteReality Preprocessing Module
---------------------------------------
This module handles the preprocessing of 3D room scans for the LiteReality pipeline.
It extracts scene data, processes object images, prepares camera data for retrieval,
and prepares the data for further processing.

Author: Zhening Huang (zh340@cam.ac.uk)
"""

import glob
import json
import os
import pickle
import shutil
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

from .object_image_extraction import (
    FrameStore,
    calculate_wall_corners,
    crop_and_save_new,
    get_pcd_from_obj,
    get_pcd_objs,
    get_pcd_objs_walls,
    get_wall_hole_points,
)
from .roomplan import (
    extract_rgbd,
    get_line_segments_and_walls,
    get_object_pose,
    get_objects,
    get_scene_usda,
    get_wall_and_object_floor_files,
    get_wall_holes,
)


def extract_ranking(filename):
    """
    Extracts the ranking number from the filename.
    
    Args:
        filename (str): The filename to extract ranking from
        
    Returns:
        int: The extracted ranking number, or 999999 if not found (to sort non-matching files last)
    """
    try:
        parts = filename.split('_')
        ranking_part = parts[-1].split('.')[0]  # Extract the ranking number before .jpg
        return int(ranking_part)
    except (ValueError, IndexError):
        # If the filename doesn't match the expected pattern, return a large number
        # to sort these files last
        return 999999


# The evidence sheet stitch_top_images writes into each object's crop folder. Named once so the
# input glob can exclude it — see the note in stitch_top_images.
STITCHED_NAME = 'stitched_image.jpg'


def stitch_top_images(folder_path, num_images=4, columns=2, gap=10):
    """
    Stitches top images from a folder into a grid layout.
    
    Args:
        folder_path (str): Path to the folder containing images
        num_images (int): Number of top images to use
        columns (int): Number of columns in the grid
        gap (int): Gap size between images in pixels
        
    Returns:
        None: Saves the stitched image to the folder
    """
    # Get all .jpg images sorted by name in the folder, EXCLUDING this function's own output.
    # `extract_ranking` sorts unmatched names last (999999), so with 4+ ranking crops the stitched
    # image fell outside the top `num_images` and was never noticed. An object with FEWER than
    # num_images crops — the wall doors/windows, which often rank only 1-3 frames — pulled the
    # previous stitched_image.jpg back into its own input, so every re-crop stitched the stitch:
    # Wall2_Door_0 went 444x1450 -> 2910x1450 -> 2910x5830 -> 11670x5830 over four runs and then
    # tripped PIL's decompression-bomb limit, at which point the object silently lost its sheet.
    image_paths = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path)
         if f.endswith('.jpg') and f != STITCHED_NAME],
        key=lambda x: extract_ranking(os.path.basename(x))
    )

    # Select the top images (up to the specified number)
    selected_images = image_paths[:num_images]
    
    # Open images, rotate them, and store their versions
    images = []
    for img_path in selected_images:
        try:
            with Image.open(img_path) as img:
                rotated_img = img.rotate(-90, expand=True)  # Rotate 90 degrees clockwise
                images.append(rotated_img)
        except Exception as e:
            print(f'{Colors.YELLOW}⚠{Colors.RESET} Warning: Failed to open image {img_path}: {e}')
            continue

    # Check if we have any valid images
    if not images:
        print(f'{Colors.YELLOW}⚠{Colors.RESET} No valid images found in {folder_path}, skipping stitching')
        return

    # Determine grid layout
    rows = (len(images) + columns - 1) // columns  # Use actual number of images, not num_images
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Resize each image to fit within the grid cell size without distortion
    resized_images = [img.resize((max_width, max_height)) for img in images]
    
    # Calculate the total size of the stitched image with gaps
    total_width = columns * max_width + (columns - 1) * gap
    total_height = rows * max_height + (rows - 1) * gap
    
    # Create a new blank image with the calculated grid size and gaps
    stitched_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    # Paste each image into the stitched image in the grid layout with gaps
    for idx, img in enumerate(resized_images):
        x_offset = (idx % columns) * (max_width + gap)
        y_offset = (idx // columns) * (max_height + gap)
        stitched_image.paste(img, (x_offset, y_offset))
    
    # Save the final stitched image as JPG (smaller file size for streaming)
    stitched_image.save(os.path.join(folder_path, STITCHED_NAME), 'JPEG', quality=85, optimize=True)
    print(f'{Colors.GREEN}✓{Colors.RESET} Stitched image saved at {folder_path}/{STITCHED_NAME}')


def process_all_folders(base_path):
    """
    Process all folders in the base path and stitch images in each folder.
    
    Args:
        base_path (str): Path to the base directory containing folders
        
    Returns:
        None
    """
    start_time = time.time()
    folder_count = 0
    
    # Iterate through each folder in the base path
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            print(f'{Colors.CYAN}📁{Colors.RESET} Processing folder: {Colors.BOLD}{folder_name}{Colors.RESET}')
            stitch_top_images(folder_path)
            folder_count += 1
    
    elapsed_time = time.time() - start_time
    print(f"{Colors.GREEN}✓{Colors.RESET} Processed {Colors.BOLD}{folder_count}{Colors.RESET} folders in {Colors.CYAN}{elapsed_time:.2f}s{Colors.RESET}")


def copy_scan_files(raw_scan_folder, name_of_scan):
    """
    Copy necessary scan files from raw scan folder to input directory.
    
    Args:
        raw_scan_folder (str): Path to the raw scan folder
        name_of_scan (str): Name identifier for the scan
        
    Returns:
        str: Path to the scene USDZ file
    """
    start_time = time.time()
    
    # Copy USDZ file
    original_usdz = os.path.join(raw_scan_folder, "roomplan", "room.usdz")
    scene_usdz = os.path.join("input", "usdz_files", f"{name_of_scan}.usdz")
    os.makedirs(os.path.dirname(scene_usdz), exist_ok=True)
    shutil.copy(original_usdz, scene_usdz)
    elapsed = time.time() - start_time
    print(f"{Colors.GREEN}✓{Colors.RESET} USDZ file copied ({Colors.CYAN}{elapsed:.2f}s{Colors.RESET})")
    
    # Copy legacy textured OBJ files into the v2 scan geometry layout.
    geometry_files_time = time.time()
    geometry_files = {
        "textured_output.obj": "scene_points.obj",
        "textured_output.mtl": "scene_points.mtl",
        "textured_output.jpg": "preview.jpg",
    }
    target_folder = os.path.join("input", "scan_geometry", name_of_scan)
    os.makedirs(target_folder, exist_ok=True)
    for src_name, dst_name in geometry_files.items():
        src = os.path.join(raw_scan_folder, src_name)
        shutil.copy(src, os.path.join(target_folder, dst_name))
    elapsed = time.time() - geometry_files_time
    print(f"{Colors.GREEN}✓{Colors.RESET} Scan geometry files copied ({Colors.CYAN}{elapsed:.2f}s{Colors.RESET})")
    
    # Extract RGBD data
    rgbd_time = time.time()
    extract_rgbd(raw_scan_folder, f"input/rgbd/{name_of_scan}")
    elapsed = time.time() - rgbd_time
    print(f"{Colors.GREEN}✓{Colors.RESET} RGBD data extracted ({Colors.CYAN}{elapsed:.2f}s{Colors.RESET})")
    
    return scene_usdz


def process_wall_holes(walls, wall_holes):
    """
    Process wall hole information from the scene.
    
    Args:
        walls (list): List of wall data
        wall_holes (dict): Dictionary of wall holes
        
    Returns:
        dict: Processed wall holes data
    """
    wall_holes_backup = wall_holes.copy()
    wall_holes_result = wall_holes.copy()
    
    for wall_name, all_wall_hole in wall_holes.items():
        if all_wall_hole:
            type_list = []
            hole_dim_list = []
            for wall_hole in all_wall_hole:
                hole_dim = wall_hole
                hole_x_1, hole_y_1, hole_x_2, hole_y_2 = hole_dim[0][0], hole_dim[0][1], hole_dim[1][0], hole_dim[1][1]
                x = (hole_x_2 - hole_x_1)
                y = (hole_y_2 - hole_y_1)
                
                # Get the center
                wall_folder = "/".join(wall_name.split("/")[:-1])
                # Get object that is not the wall
                wall_hole_files = glob.glob(f"{wall_folder}/*")
                wall_hole_files = [file for file in wall_hole_files if "Wall" not in file.split("/")[-1]]
                
                if wall_hole_files == []:
                    wall_holes_result[wall_name] = []
                    continue
                
                type_name = {}
                for hole_file in wall_hole_files:
                    name = hole_file.split("/")[-1].split(".")[0]
                    _, hole = get_line_segments_and_walls([hole_file])
                    type_name[name] = hole
                
                wall_holes[wall_name] = {
                    "type": type_name,
                    "dimension": wall_hole
                }
                
                hole_type_info = {}
                for key, value in wall_holes[wall_name]["type"].items():
                    hole_type_info[key] = [value[0]["pose"]["bbox"][:2]]
                
                # Select the closest one based on both s and w
                distances = []
                for key, value in hole_type_info.items():
                    distances.append(np.linalg.norm(np.array(value) - np.array([x, y])))

                type_key = list(hole_type_info.keys())[np.argmin(distances)]
                type_simply = "Door" if "Door" in type_key else "Window"

                type_list.append(type_simply)
                hole_dim_list.append(hole_dim)

                wall_holes_result[wall_name] = {
                    "type": type_list,
                    "dimension": hole_dim_list
                }
    
    return wall_holes_result


def extract_scene_data(scene_usdz):
    """
    Extract and process scene data from USDZ file.
    
    Args:
        scene_usdz (str): Path to the scene USDZ file
        
    Returns:
        tuple: Contains walls, objects, wall_holes_result, and floor data
    """
    start_time = time.time()
    
    # Convert USDZ to USDA
    scene_usda_path = get_scene_usda(scene_usdz)
    
    # Extract wall, objects and floor files
    wall_and_object_files = get_wall_and_object_floor_files(room_usda=scene_usda_path)
    wall_files = wall_and_object_files["walls"]
    object_files = wall_and_object_files["objects"]
    floor_file = wall_and_object_files["Floor"]
    
    # Get walls, objects, and floor data
    _, walls = get_line_segments_and_walls(wall_files)
    objects = get_objects(object_files)
    floor = get_object_pose(wall_and_object_files["Floor"][0])
    
    # Extract wall holes
    wall_holes = {wall["file"]: get_wall_holes(wall["pose"]) for wall in walls}
    
    # Process wall holes
    wall_holes_result = process_wall_holes(walls, wall_holes)
    
    elapsed = time.time() - start_time
    print(f"{Colors.GREEN}✓{Colors.RESET} Scene data extracted ({Colors.CYAN}{elapsed:.2f}s{Colors.RESET})")
    return walls, objects, wall_holes_result, floor


def save_scene_data(name_of_scan, walls, objects, wall_holes_result, floor):
    """
    Save the preprocessed scene data to files.
    
    Args:
        name_of_scan (str): Name identifier for the scan
        walls (list): Wall data
        objects (list): Object data
        wall_holes_result (dict): Wall holes data
        floor (dict): Floor data
        
    Returns:
        str: Path to the output folder
    """
    start_time = time.time()
    
    output_folder = f"input/scene_data/{name_of_scan}"
    os.makedirs(output_folder, exist_ok=True)

    for data, name in zip(
        [walls, objects, wall_holes_result, floor], 
        ["walls", "objects", "wall_holes", "floor"]
    ):
        with open(f"{output_folder}/{name}.pkl", "wb") as f:
            pickle.dump(data, f)
    
    elapsed = time.time() - start_time
    print(f"{Colors.GREEN}✓{Colors.RESET} Scene data saved ({Colors.CYAN}{elapsed:.2f}s{Colors.RESET})")
    return output_folder


def process_object_images(
    name_of_scan,
    walls,
    objects,
    wall_holes_result,
    include_walls=False,
    enlarged_crop_objects=(),
):
    """
    Process and extract 2D images for objects from point cloud.
    
    Args:
        name_of_scan (str): Name identifier for the scan
        walls (list): Wall data
        objects (list): Object data
        wall_holes_result (dict): Wall holes data
        include_walls (bool): Whether to save crop references for each full wall
        
    Returns:
        None
    """
    start_time = time.time()
    enlarged_crop_objects = set(enlarged_crop_objects)
    print(f"{Colors.BLUE}🖼️{Colors.RESET} Starting to obtain 2D images for objects...")
    
    # Update wall corners data
    for wall in walls:
        position = wall["pose"]["position"]
        rotation = wall["pose"]["rotation"]
        bbox = wall["pose"]["bbox"]
        wall_corners = calculate_wall_corners(position, rotation, bbox)
        wall["pose"]["top_down_rect"] = wall_corners.tolist()

    # Setup paths
    scene_pcd_path = f"input/scan_geometry/{name_of_scan}/scene_points.obj"
    if not os.path.exists(scene_pcd_path):
        raise FileNotFoundError(scene_pcd_path)
    scene_data_path = f"input/scene_data/{name_of_scan}/objects.pkl"
    image_dir = f"input/rgbd/{name_of_scan}"
    parsed_image_dir = f"input/parsed_images/{name_of_scan}"

    # Load objects data and scene point cloud
    with open(scene_data_path, "rb") as f:
        objects = pickle.load(f)
    pcd_scene = get_pcd_from_obj(scene_pcd_path)

    # Get object and wall point clouds
    bbox, crop_out_objs = get_pcd_objs(objects, pcd_scene)
    bbox_walls, crop_out_objs_walls, rotation = get_pcd_objs_walls(walls, pcd_scene)

    # Process wall holes
    wall_holes_rename = wall_holes_result.copy()
    for key in list(wall_holes_rename.keys()):
        new_name = key.split("/")[-1].split(".")[0]
        wall_holes_rename[new_name] = wall_holes_rename.pop(key)

    crop_out_objs_windows = get_wall_hole_points(wall_holes_rename, rotation, bbox_walls, pcd_scene)

    # Extract floor point cloud
    floor_cropped_mask = pcd_scene[:, 1] < (pcd_scene[:, 1].min() + 0.3)
    floor_pcd = pcd_scene[floor_cropped_mask]
    crop_out_objs["Floor"] = floor_pcd

    if include_walls:
        crop_out_objs = {**crop_out_objs, **crop_out_objs_walls}

    # Merge the objects and wall-hole dict
    crop_out_objs = {**crop_out_objs, **crop_out_objs_windows}

    # Prepare bbox dictionary
    bbox_dict = bbox.copy()
    if include_walls:
        bbox_dict.update(bbox_walls)
    bbox_dict["Floor"] = np.zeros((8, 3))
    
    # Every object ranks itself against the SAME frames, so the capture is loaded once here and
    # shared, rather than re-read inside each object's frame sweep. See FrameStore.
    frames = FrameStore(name_of_scan)

    # Process each object
    for object_id, pcd_input in tqdm(crop_out_objs.items()):
        obj_start_time = time.time()
        if "Wall" in object_id and "_" in object_id:
            bbox_dict[object_id] = np.zeros((8, 3))
        if len(pcd_input) == 0:
            print(f"  {Colors.YELLOW}⚠{Colors.RESET} Warning: No visible points for {object_id}, skipping")
            continue

        top_k_images = crop_and_save_new(image_dir, pcd_input, object_id, bbox_dict.get(object_id, np.zeros((8, 3))), top_k=4, frames=frames)
        if not top_k_images:
            print(f"  {Colors.YELLOW}⚠{Colors.RESET} Warning: No ranked crops for {object_id}, skipping")
            continue

        # Define save paths for images and camera data
        object_save_dir = os.path.join(parsed_image_dir, object_id)
        camera_data_dir = os.path.join(object_save_dir, "camera")
        os.makedirs(object_save_dir, exist_ok=True)
        os.makedirs(camera_data_dir, exist_ok=True)

        for i, (frame_id, image_info) in enumerate(top_k_images):
            # Load image and crop bounding box
            image_path = os.path.join(image_dir, "image", f"frame_{frame_id}.jpg")
            image = cv2.imread(image_path)
            
            bbox = image_info["bbox"]
            # ENLARGED CROP: for merged sets (e.g. a sink + its base-cabinet "Sink_Storage" unit),
            # the visible-points bbox only covers the well-scanned part (the sink). Crop instead to
            # the full projected 3D box so the evidence — and the generated reference — captures the
            # WHOLE unit. Callers opt in per object; a small margin guards edges.
            if object_id in enlarged_crop_objects and image_info.get("bbox3d"):
                b = image_info["bbox3d"]
                mx = int((b[2] - b[0]) * 0.06); my = int((b[3] - b[1]) * 0.06)
                bbox = [max(0, b[0] - mx), max(0, b[1] - my), min(256, b[2] + mx), min(192, b[3] + my)]
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
                "resized_bbox": resized_bbox,
                "ranking": i
            }

            # Save cropped image and metadata
            cv2.imwrite(os.path.join(object_save_dir, f"frame_{frame_id}_ranking_{str(i)}.jpg"), cropped_image)
            with open(os.path.join(camera_data_dir, f"frame_{frame_id}_ranking_{str(i)}.json"), "w") as f:
                json.dump(frame_info, f)
        
        elapsed = time.time() - obj_start_time
        print(f"  {Colors.GREEN}✓{Colors.RESET} Processed {Colors.BOLD}{object_id}{Colors.RESET} ({Colors.CYAN}{elapsed:.2f}s{Colors.RESET})")

    elapsed = time.time() - start_time
    print(f"{Colors.GREEN}✓{Colors.RESET} Object image processing completed ({Colors.CYAN}{elapsed:.2f}s{Colors.RESET})")


def extract_object_id(obj_path):
    """
    Extract object ID from the object file path.
    
    Args:
        obj_path (str): Path to the object directory
        
    Returns:
        str: Object ID
    """
    return obj_path.split("/")[-1]


def extract_scene_name(path):
    """
    Extract scene name from the path.
    
    Args:
        path (str): Path containing scene name
        
    Returns:
        str: Scene name
    """
    return path.split("/")[-1]


def create_output_directory(scene_name, obj_id):
    """
    Create output directory for storing retrieval images and data.
    
    Args:
        scene_name (str): Name of the scene
        obj_id (str): Object ID
        
    Returns:
        str: Path to the output directory
    """
    output_dir = f"input/object_stage/{scene_name}/{obj_id}/images"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def process_camera_model(camera_model_path, output_dir):
    """
    Process a single camera model JSON file.
    
    Args:
        camera_model_path (str): Path to the camera model JSON file
        output_dir (str): Directory to store the images
        
    Returns:
        tuple: (image_name, data) containing processed information
    """
    with open(camera_model_path) as f:
        data = json.load(f)
    
    # Copy the original image to the output directory
    image_path = data["original_image_path"]
    shutil.copy(image_path, output_dir)
    
    # Extract image name without extension
    image_name = str(image_path.split("/")[-1].split(".")[0])
    
    return image_name, data


def process_object_for_retrieval(obj_path, scene_name):
    """
    Process a single object directory to extract camera and bbox data for retrieval.
    
    Args:
        obj_path (str): Path to the object directory
        scene_name (str): Name of the scene
        
    Returns:
        None
    """
    obj_id = extract_object_id(obj_path)
    
    # Find all camera model JSON files
    camera_model_folder = f"{obj_path}/camera/*.json"
    camera_models = glob.glob(camera_model_folder)
    
    # Create output directory
    output_dir = create_output_directory(scene_name, obj_id)
    
    # Initialize data dictionaries
    bbox_info_sum = {}
    camera_pose_sum = {}
    data = None  # Initialize data to None to prevent UnboundLocalError
    
    # Process each camera model
    for camera_model in camera_models:
        image_name, data = process_camera_model(camera_model, output_dir)
        
        # Extract bounding box information
        bbox_info = data["resized_bbox"]
        bbox_info_sum[image_name] = bbox_info
        
        # Extract camera pose information
        camera_pose_sum[image_name] = {
            "intrinsic": data["intrinsic"],
            "pose": data["pose"],
            "bbox": data["bbox"],
            "dimensions": [256, 192],
        }
    
    # Check if we have any camera models processed
    if not camera_models or data is None:
        print(f'{Colors.YELLOW}⚠{Colors.RESET} Warning: No camera models found for {obj_id}, skipping')
        return
    
    # Add semantic information
    bbox_info_sum["semantic"] = data["semantic"]
    
    # Save data to JSON files
    with open(f"{output_dir}/../bbox_info.json", "w") as f:
        json.dump(bbox_info_sum, f)
    with open(f"{output_dir}/../camera_pose_info.json", "w") as f:
        json.dump(camera_pose_sum, f)


def prepare_camera_data_for_retrieval(name_of_scan):
    """
    Prepare camera data from parsed images for the retrieval stage.
    This organizes camera information, bounding boxes, and images for object retrieval.
    
    Args:
        name_of_scan (str): Name identifier for the scan
        
    Returns:
        None
    """
    start_time = time.time()
    parsed_image_dir = f"input/parsed_images/{name_of_scan}"
    
    # Get all object directories
    all_objs = glob.glob(f"{parsed_image_dir}/*")
    scene_name = name_of_scan
    
    print(f"{Colors.BLUE}📷{Colors.RESET} Starting camera data preparation for scene: {Colors.BOLD}{scene_name}{Colors.RESET}")
    print(f"{Colors.CYAN}  Found {Colors.BOLD}{len(all_objs)}{Colors.RESET}{Colors.CYAN} objects to process{Colors.RESET}")
    
    # Process each object with progress bar
    for obj_path in tqdm(all_objs, desc="Processing objects", colour='cyan'):
        if os.path.isdir(obj_path):
            process_object_for_retrieval(obj_path, scene_name)
    
    elapsed_time = time.time() - start_time
    print(f"{Colors.GREEN}✓{Colors.RESET} Camera data preparation completed ({Colors.CYAN}{elapsed_time:.2f}s{Colors.RESET})")


def main():
    """Main function to execute the preprocessing pipeline."""
    # Set up argument parser
    import argparse
    parser = argparse.ArgumentParser(description="LiteReality Pipeline")
    
    # Define arguments
    parser.add_argument("--raw", type=str, default="data/scans/2024_10_23_15_37_30", 
                        help="Path to the folder containing raw scan data.")
    parser.add_argument("--name", type=str, default="joan", 
                        help="Name identifier for the scan.")
    
    # Parse arguments
    args = parser.parse_args()
    raw_scan_folder = args.raw
    name_of_scan = args.name

    total_start_time = time.time()
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}║{Colors.RESET} {Colors.BOLD}{Colors.CYAN}🚀 LiteReality Preprocessing Pipeline{Colors.RESET} {Colors.BOLD}{Colors.MAGENTA}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"{Colors.CYAN}Started at:{Colors.RESET} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Colors.CYAN}Scene:{Colors.RESET} {Colors.BOLD}{name_of_scan}{Colors.RESET}\n")
    
    # Step 1: Copy scan files
    print(f"{Colors.BOLD}{Colors.BLUE}[Step 1/6]{Colors.RESET} Copying scan files...")
    scene_usdz = copy_scan_files(raw_scan_folder, name_of_scan)
    
    # Step 2: Extract scene data
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 2/6]{Colors.RESET} Extracting scene data...")
    walls, objects, wall_holes_result, floor = extract_scene_data(scene_usdz)
    
    # Step 3: Save scene data
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 3/6]{Colors.RESET} Saving scene data...")
    save_scene_data(name_of_scan, walls, objects, wall_holes_result, floor)
    
    # Step 4: Process object images
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 4/6]{Colors.RESET} Processing object images...")
    process_object_images(name_of_scan, walls, objects, wall_holes_result)
    
    # Step 5: Stitch images
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 5/6]{Colors.RESET} Stitching images...")
    stitch_time = time.time()
    parsed_image_dir = f"input/parsed_images/{name_of_scan}"
    process_all_folders(parsed_image_dir)
    elapsed = time.time() - stitch_time
    print(f"{Colors.GREEN}✓{Colors.RESET} Image stitching completed ({Colors.CYAN}{elapsed:.2f}s{Colors.RESET})")
    
    # Step 6: Prepare camera data for retrieval
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 6/6]{Colors.RESET} Preparing camera data for retrieval...")
    prepare_camera_data_for_retrieval(name_of_scan)
    
    total_time = time.time() - total_start_time
    print(f"\n{Colors.BOLD}{Colors.GREEN}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}║{Colors.RESET} {Colors.BOLD}✓ Preprocessing completed successfully!{Colors.RESET} {Colors.BOLD}{Colors.GREEN}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"{Colors.CYAN}Total time:{Colors.RESET} {Colors.BOLD}{total_time:.2f}s{Colors.RESET}\n")


if __name__ == "__main__":
    main()
