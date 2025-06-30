import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from superpoint.datasets.colon import Colon
from superpoint.datasets.utils import pipeline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Disable eager execution
tf.compat.v1.disable_eager_execution()

def visualize_mask_for_image(config, image_path, camera_mask_path, output_path=None):
    """
    Visualize masks for a specific image with robust error handling.
    """
    # Reset the default graph to avoid conflicts
    tf.reset_default_graph()
    
    # Create a TensorFlow session
    with tf.Session() as sess:
        try:

            print("Creating Colon instance...")
            colon = Colon(**config)

            print("Loading and preprocessing image...")
            # # Create image tensor
            img_tensor = tf.read_file(image_path)
            img_tensor = tf.image.decode_png(img_tensor, channels=3)
            img_tensor = tf.cast(img_tensor, tf.float32)
            img_tensor = img_tensor / 255.0
            img_tensor = tf.image.rgb_to_grayscale(img_tensor)
            img_tensor = pipeline.colonoscopy_preprocess(img_tensor, **config["preprocessing"])

            img_value = sess.run(img_tensor)
            print(f"Image tensor shape: {img_value.shape}, min: {np.min(img_value)}, max: {np.max(img_value)}")


            print("Loading camera mask...")
            # Load camera mask - run this separately to identify issues
            camera_mask = colon._load_camera_mask(camera_mask_path)
            camera_mask = pipeline.colonoscopy_preprocess(camera_mask, **config["preprocessing"])
            camera_mask_value = sess.run(camera_mask)
            print(f"Camera mask shape: {camera_mask_value.shape}, min: {np.min(camera_mask_value)}, max: {np.max(camera_mask_value)}")

            print("Generating specular mask...")
            # Generate specular mask - run separately

            specular_mask = colon._generate_specular_mask(img_tensor)
            specular_mask_value = sess.run(specular_mask)
            print(f"Specular mask shape: {specular_mask_value.shape}, sum: {np.sum(specular_mask_value)}")
            
            print("Computing combined mask...")
            # Generate combined mask - run separately
            combined_mask = colon._compute_combinated_mask(camera_mask, specular_mask)
            combined_mask_value = sess.run(combined_mask)
            print(f"Combined mask shape: {combined_mask_value.shape}, sum: {np.sum(combined_mask_value)}")
            
            # Visualization works with the values we already computed
            img_np = img_value
            camera_np = camera_mask_value
            specular_np = specular_mask_value
            combined_np = combined_mask_value
            
        except Exception as e:
            print(f"Error during mask generation: {str(e)}")
            print("Entering debugging mode with PDB. Type 'c' to continue or 'q' to quit.")
            import pdb;pdb.set_trace()
            raise e
    
    print("Generating visualization plot...")
    # Visualize with matplotlib (outside the TensorFlow session)
    plt.figure(figsize=(16, 12))
    
    plt.subplot(221)
    plt.imshow(img_np.squeeze(), cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(222)
    plt.imshow(camera_np.squeeze(), cmap='gray')
    plt.title('Camera Mask (Black = Invalid)')
    
    plt.subplot(223)
    plt.imshow(specular_np.squeeze(), cmap='gray')
    plt.title('Specular Mask (Black = Invalid)')
    
    plt.subplot(224)
    plt.imshow(combined_np.squeeze(), cmap='gray')
    plt.title('Combined Mask (Black = Invalid)')
    
    plt.tight_layout()
    
    if output_path:
        # Create directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Now save the figure
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    # Replace these with your actual paths
    IMAGE_PATH = "/home/student/ColonSuperPoint/ColonExperiments/datasets/endomapper_train/33/images/out04066.png"  
    CAMERA_MASK_PATH = "/home/student/ColonSuperPoint/ColonExperiments/datasets/endomapper_train/33/camera_mask.png"
    OUTPUT_PATH = "/home/student/ColonSuperPoint/ColonExperiments/datasets/endomapper_train/33/masks/visualization.png"
    
    config = {
    'image_path': '/home/student/ColonSuperPoint/ColonExperiments/datasets/endomapper_train/33/images',
    'preprocessing':{
        'use_colonoscopy_preprocess': True,
        'half_resolution': False,
        'camera_mask_path': '/home/student/ColonSuperPoint/ColonExperiments/datasets/endomapper_train/33/camera_mask.png',
    },
    'truncate': 50,
}

    visualize_mask_for_image(config, IMAGE_PATH, CAMERA_MASK_PATH, OUTPUT_PATH)