import os
import random
from pathlib import Path
import numpy as np
import pycolmap
import shutil

def set_all_seeds(seed):
    """Set all seeds for reproducible results."""

    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Set environment variable for any libraries that check it
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"All seeds set to {seed} for reproducibility")
    
def subsample_images(input_dir, output_dir, num_images=500):
    """
    Subsamples a specified number of images from a dataset directory and saves them to an output directory.
    
    Args:
        input_dir (str): Path to the input dataset directory containing images.
        output_dir (str): Path to the output directory where subsampled images will be saved.
        num_images (int): Number of images to subsample. Default is 500.
    """
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all image files in the input directory
    image_files = list(Path(input_dir).glob('*.png'))  # Adjust the extension as needed
    
    # Randomly select a subset of images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    # Copy selected images to the output directory
    for img_file in selected_images:
        dest_path = Path(output_dir) / img_file.name
        shutil.copy2(img_file, dest_path)  # Use copy2 to preserve metadata
    
    print(f"Subsampled {len(selected_images)} images to {output_dir}")
    return selected_images

def main():
    input_dir = "/home/student/ColonSuperPoint/datasets/dataset_001_002/images/"
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    output_dir = "/home/student/ColonSuperPoint/datasets/dataset_001_002/images_subsampled/"
    subsample_images(input_dir, output_dir)

if __name__ == "__main__":
    set_all_seeds(42)  # Set seed for reproducibility
    main()