# Enseures covisibility between images in the dataset.
import os
import numpy as np
from pathlib import Path
import pycolmap
import matplotlib.pyplot as plt
import torch
import random
import cv2

###### Functions ######
def set_all_seeds(seed):
    """Set all seeds for reproducible results."""

    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for any libraries that check it
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"All seeds set to {seed} for reproducibility")


def load_covisibility_graph(file_path):
    """
    Loads a covisibility graph from a text file.
    The covisibility graph is represented as a dictionary where each key is a reference image
    and its value is a list of covisible images. The input file should have each line formatted
    as follows:
        reference_image covisible_image_1 covisible_image_2 ...
    If a reference image has no covisible images, the line should only contain the reference image.
    Args:
        file_path (str): The path to the text file containing the covisibility graph.
    Returns:
        dict: A dictionary representing the covisibility graph. Keys are reference image names
              (str), and values are lists of covisible image names (list of str).
    """

    covisibility_graph = {}

    with open(file_path, 'r') as f:
        for line in f: 
            line = line.strip().split()
            if not line:
                continue
    
            reference_image = line[0]
            covisible_images = line[1:] if len(line) > 1 else []
            covisibility_graph[reference_image] = covisible_images
    
    return covisibility_graph

def get_img_name(img_path):
    """
    Get the image name considering the parent folder structure, where
    the parent folder is either "exterior" or "interior" (like Graham Hall dataset).
    Other labels could be considered for other datasets.
    
    Args:
        img_path: Path object of the image
        
    Returns:
        str: Formatted image name
    """
    uproot_folder = img_path.parent.name
    
    # Check if the parent folder is "exterior" or "interior"
    if uproot_folder in ["exterior", "interior"]:
        # Add the parent folder of the image to the image name
        return f"{uproot_folder}/{img_path.name}"
    else:
        return img_path.name
    
def filter_screenshot(images, images_dir, threshold=20):
    """
    Filter out images that have a thumbnail/doctor interface in bottom-left corner.
    """
    filtered_images = []
    problem_images = []
    
    # Rectangle coordinates for bottom-left corner artifact
    x1, y1 = 0, 820    # Top-left
    x2, y2 = 25, 1020  # Bottom-right
    
    for img_name in images:
        img_path = images_dir / img_name
        try:
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"Failed to load image {img_path}")
                problem_images.append(img_name)
                continue
                
            # Add dimension check
            if y2 > img.shape[0] or x2 > img.shape[1]:
                print(f"Warning: ROI outside image bounds for {img_name}. Image shape: {img.shape}")
                # Adjust ROI to fit within image
                y2_adj = min(y2, img.shape[0])
                x2_adj = min(x2, img.shape[1])
                roi = img[y1:y2_adj, x1:x2_adj]
            else:
                roi = img[y1:y2, x1:x2]
         
            # Convert to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Count non-black pixels
            non_black_pixels = np.sum(gray_roi > threshold)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            # If less than 5% of pixels are non-black, keep the image
            percentage = (non_black_pixels / total_pixels) * 100
            if percentage < 5:
                filtered_images.append(img_name)
            else:
                print(f"Filtering out {img_name} - corner artifact detected ({percentage:.2f}% non-black)")
                
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            problem_images.append(img_name)
            
    print(f"Filtered {len(images) - len(filtered_images) - len(problem_images)} images with corner artifacts")
    print(f"Failed to process {len(problem_images)} images")
    return filtered_images
#######################


############################# Seq #############################
seq = "seq_001"
# seq = "seq_002"

seq_dir = Path(f'/home/student/colon_matching/data/{seq}')
sparse_dir = seq_dir / 'sparse'
###############################################################

# Set seeds for reproducibility
seed = 42
set_all_seeds(seed)

# Get all submaps of sequence get the names of the submaps
submaps = [int(f.stem) for f in sparse_dir.iterdir() if f.is_dir()]
# Sort in ascending order
submaps.sort()
# Convert to string
submaps = [str(submap) for submap in submaps]

# Iterate over all the submaps in the sequence
for submap in submaps:
    """
    submaps to avoid:
    - 102, 106, 107, 109, 113, 115, 131, 210, 221, 240
    """

    if seq == "seq_001" and submap in ['2','6','7','9', '13', '15','31']:
        continue
    if seq == "seq_002" and submap in ['10','21','40']:
        continue

    # Define the paths for the submap model and the source of images of the mode
    sparse_model_dir = seq_dir / 'sparse' / submap
    images_dir = seq_dir / 'submap_images' / submap
    # images_dir = seq_dir / 'toy_for_artifact' / submap

    print(f"Processing submap {submap} in sequence {seq}...")
    
    # Load the COLMAP reconstruction for the submap
    reconstruction = pycolmap.Reconstruction(str(sparse_model_dir))

    #Get image pairs for the submap
    images = list(images_dir.glob('*.png'))

    # Load previously generated covisibility graph (generate_cameras_model.py)
    covisibility_path = sparse_model_dir / "camerasModel.txt"
    covisibility_graph = load_covisibility_graph(covisibility_path)
    # Sort images to ensure they're in sequential order
    images.sort(key=lambda x: x.name)

    # Initialize filtered list 
    filtered_list = []

    # Select only images that are present in the covisibility graph
    for i in range(len(images)):
        img = images[i]
        img = get_img_name(img)
        if img not in covisibility_graph:
            print(f"Image {img} not found in covisibility graph for submap {submap}.")
            continue
        filtered_list.append(img)

    final_images = filter_screenshot(filtered_list, images_dir)

    # Copy the images to new dataset folder
    new_images_dir = f"/home/student/ColonSuperPoint/datasets/dataset_001_002/images/"
    os.makedirs(new_images_dir, exist_ok=True)
    for img in final_images:
        src_img_path = seq_dir / 'submap_images' / submap / img
        dst_img_path = Path(new_images_dir) / f"{seq}_{img}"
        dst_img_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.system(f"cp {src_img_path} {dst_img_path}")
        except Exception as e:
            print(f"Error copying {src_img_path} to {dst_img_path}: {e}")

