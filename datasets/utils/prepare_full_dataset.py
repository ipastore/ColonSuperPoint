import cv2
import os
import numpy as np

# Input and output directories
input_dir = "extracted_frames"  # Directory where extracted frames are stored
output_dir = "filtered_frames"  # Directory to save filtered frames
os.makedirs(output_dir, exist_ok=True)

# Thresholds for filtering
brightness_threshold = 220  # Adjust based on your dataset (higher for brighter images)
texture_threshold = 50      # Laplacian variance threshold (lower for less texture)

def is_low_texture(image):
    """Check if the image has low texture based on Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < texture_threshold

def is_overexposed(image):
    """Check if the image is overly bright based on mean brightness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness > brightness_threshold

# Process and filter each frame
for frame_name in os.listdir(input_dir):
    if frame_name.endswith(".png"):  # Process only .png images
        frame_path = os.path.join(input_dir, frame_name)
        image = cv2.imread(frame_path)

        # Apply filtering conditions
        if not is_low_texture(image) and not is_overexposed(image):
            # Save the frame if it passes both filters
            output_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(output_path, image)
            print(f"Saved: {frame_name}")
        else:
            print(f"Filtered out: {frame_name} (low texture or too bright)")