import numpy as np
import tensorflow as tf
from pathlib import Path

from .base_dataset import BaseDataset
from .utils import pipeline
from superpoint.settings import DATA_PATH, EXPER_PATH

class Colon(BaseDataset):
    default_config = {
        'labels': None,
        'cache_in_memory': False,
        'validation_size': 0,
        'truncate': None,
        'preprocessing': {
            'use_colonoscopy_preprocess': True,
            'half_resolution': False,  # Set to true when you want half resolution
            'camera_mask_path': '/home/student/ColonSuperPoint/ColonExperiments/datasets/endomapper_train/33/camera_mask.png',
        },
        'num_parallel_calls': 10,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
    }

    def _load_camera_mask(self, mask_path, kernel_size=5, iterations=3):
        """
        Load fixed camera mask using TensorFlow, avoiding the black corners of the colonoscopy images.
        
        Args: 
            mask_path (str): Path to the camera mask file. Where the corners are WHITE and the rest is BLACK.
            kernel_size (int): Size of the kernel for erosion. Default is 5.
            iterations (int): Number of erosion iterations to apply. Default is 3.

        Returns: A float32 mask for the Endomapper cameras, masking out the black corners. Range [0,1].
        """
        # Ensure the mask path exists
        assert mask_path is not None and Path(mask_path).exists(), "Mask path must be provided."

        # Read the mask file
        mask = tf.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)  # Load as grayscale (1 channel)

        # Assert that max of mask is > 1
        max_assertion = tf.debugging.assert_greater(
            tf.reduce_max(mask), tf.constant(1, dtype=mask.dtype), message="Mask should be in [0,255] range."
        )
        
        with tf.control_dependencies([max_assertion]):
            
            # Convert to boolean
            mask = mask < 127

            # Convert to 4D for dilation2d: [1, H, W, 1]
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=0)  # Add batch
            # mask = tf.expand_dims(mask, axis=-1)  # Add channel

            kernel = tf.zeros((kernel_size, kernel_size, 1), dtype=tf.float32)

            # Apply erosion for n iterations
            def erode_iteration(i, current_mask):
                eroded = tf.nn.erosion2d(
                    current_mask,
                    kernel,
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding='SAME'
                )
                return i + 1, eroded
            
            # TensorFlow while loop for iterations
            _, eroded_mask = tf.while_loop(
                lambda i, mask: i < iterations,
                erode_iteration,
                [tf.constant(0), mask],
                parallel_iterations=1
            )

            # Squeeze back to [H, W] and threshold
            mask = tf.squeeze(eroded_mask, axis=[0])  # [H, W, C]
            # mask = tf.cast(mask , tf.float32)  # Convert to float32
            mask = tf.identity(mask)  # Ensure the mask is evaluated within control dependencies

            return mask

    def _generate_specular_mask(self, image, threshold=0.86, kernel_size=5, iterations=10):
        """ 
        Generate a specularity mask for the image.
        
        Args: 
            image (tf.Tensor): Input image tensor of shape [H, W, C]. Range [0, 1].
            threshold (float): Threshold to determine specularities. Default is 0.86 (220/255).
            kernel_size (int): Size of the kernel for dilation. Default is 5.
            iterations (int): Number of erosion iterations. Default is 10.

        Returns: A boolean mask where Black indicate specularities. Masking out overexposed pixels.
        """
        shape_assertion = tf.debugging.assert_equal(
            tf.rank(image), 3, message="Image must have 3 dimensions: [H, W, C]"
        )

        with tf.control_dependencies([shape_assertion]):
            # Check if normalization is needed
            max_value = tf.reduce_max(image)
            normalized_image = tf.cond(
                max_value > 1.0,
                lambda: image / 255.0,
                lambda: image
            )

            # Remove the C dimension
            image_gray = tf.squeeze(normalized_image, axis=-1)
            
            # Create binary mask of over-exposed pixels
            over_exposed_mask = image_gray < threshold  # tf.bool, [H, W]

            # Convert to 4D for dilation2d: [1, H, W, 1]
            over_exposed_mask_4d = tf.cast(over_exposed_mask, tf.float32)
            over_exposed_mask_4d = tf.expand_dims(over_exposed_mask_4d, axis=0)  # Add batch
            over_exposed_mask_4d = tf.expand_dims(over_exposed_mask_4d, axis=-1)  # Add channel

            # Define the dilation kernel
            kernel = tf.zeros((kernel_size, kernel_size, 1), dtype=tf.float32)

            # Apply erosion multiple times using a loop
            def erode_iteration(i, current_mask):
                eroded = tf.nn.erosion2d(
                    current_mask,
                    kernel,
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding='SAME'
                )
                return i+1, eroded
                
            # TensorFlow while loop for iterations
            _, eroded_mask = tf.while_loop(
                lambda i, mask: i < iterations,
                erode_iteration,
                [tf.constant(0), over_exposed_mask_4d],
                parallel_iterations=1
            )

            # Squeeze back to [H, W] and threshold
            eroded_mask = tf.squeeze(eroded_mask, axis=[0, -1])  # [H, W]
            eroded_mask = tf.cast(eroded_mask > 0, tf.bool)

            # Final specular *mask out* → we return False where specularities are
            # final_mask = tf.logical_not(eroded_mask)  # [H, W]

            final_mask = tf.identity(eroded_mask)  # Ensure the mask is evaluated within control dependencies

            return final_mask

    def _compute_combinated_mask(self, camera_mask, specular_mask):
        """ Combine the camera mask and the specular mask

        Args:
            camera_mask (tf.Tensor): Float32 mask for the camera, where Black indicates corners.
            specular_mask (tf.Tensor): Boolean mask for specularities, where Black indicates specularities.
        
        Returns:
            A Boolean mask that combines both masks, where Black indicates both corners and specularities.
            The idea is to mask out the invalid regions.
        """

        type_assertion1 = tf.debugging.assert_type(camera_mask, tf.float32, message="Camera mask must be a float32 tensor.")
        type_assertion2 = tf.debugging.assert_type(specular_mask, tf.bool, message="Specular mask must be a boolean tensor.")
       
        # Remove C dimension from camera mask
        camera_mask = tf.squeeze(camera_mask, axis=-1)

        shape_assertion = tf.debugging.assert_equal(
            tf.shape(camera_mask), tf.shape(specular_mask),
            message="Camera mask and specular mask must have the same shape."
        )

        # Use control dependencies to ensure assertions are evaluated
        with tf.control_dependencies([type_assertion1, type_assertion2, shape_assertion]):
            # Convert camera mask to boolean
            camera_mask = tf.cast(camera_mask > 0.5, tf.bool)
            
            # Combine the masks using logical AND
            combined_mask = tf.logical_and(camera_mask, specular_mask)
            
            # Important: Identity op inside control dependencies ensures assertions run
            combined_mask = tf.identity(combined_mask)

            return combined_mask


    def _init_dataset(self, **config):

        # Override the base path if specified in config
        if 'image_path' in config:
            base_path = Path(config['image_path'])
        else:
            base_path = Path(DATA_PATH, 'endomapper_train/33/images/')

        image_paths = list(base_path.iterdir())
        if config['truncate']:
            image_paths = image_paths[:config['truncate']]
        names = [p.stem for p in image_paths]
        image_paths = [str(p) for p in image_paths]
        files = {'image_paths': image_paths, 'names': names}

        if config['labels']:
            label_paths = []
            for n in names:
                p = Path(EXPER_PATH, config['labels'], '{}.npz'.format(n))
                assert p.exists(), 'Image {} has no corresponding label {}'.format(n, p)
                label_paths.append(str(p))
            files['label_paths'] = label_paths

        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
                fn, num_parallel_calls=config['num_parallel_calls'])

        return files

    def _get_data(self, files, split_name, **config):
        has_keypoints = 'label_paths' in files
        is_training = split_name == 'training'

        def _read_image(path):
            image = tf.read_file(path)
            image = tf.image.decode_png(image, channels=3)
            return tf.cast(image, tf.float32)
        

        # SP-colon: preprocess with Rodriguez code
        def _preprocess(image):
            image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing'].get('use_colonoscopy_preprocess', True):
                image = pipeline.colonoscopy_preprocess(image, **config['preprocessing'])
            elif config['preprocessing'].get('resize'):
                image = pipeline.ratio_preserving_resize(image, **config['preprocessing'])
            return image

        # Python function
        def _read_points(filename):
            return np.load(filename.decode('utf-8'))['points'].astype(np.float32)
        
        if config['preprocessing'].get('use_colonoscopy_preprocess', True):
            assert 'camera_mask_path' in config['preprocessing'], "Camera mask path must be provided in the config, under preprocessing"

        names = tf.data.Dataset.from_tensor_slices(files['names'])
        images_paths = tf.data.Dataset.from_tensor_slices(files['image_paths'])
        images = images_paths.map(_read_image)
        images = images.map(_preprocess)
        

        
        camera_mask_path = config['preprocessing'].get('camera_mask_path')
        def _add_mask(image):

            camera_mask = self._load_camera_mask(camera_mask_path)
            camera_mask = pipeline.colonoscopy_preprocess(
                camera_mask, **config['preprocessing'])
            specular_mask = self._generate_specular_mask(image)
            mask = self._compute_combinated_mask(camera_mask, specular_mask)
            return mask

        masks = images.map(_add_mask)
        data = tf.data.Dataset.zip({'image': images, 'mask': masks, 'name': names})

        # Add keypoints
        if has_keypoints:
            kp = tf.data.Dataset.from_tensor_slices(files['label_paths'])
            kp = kp.map(lambda path: tf.py_func(_read_points, [path], tf.float32))
            kp = kp.map(lambda points: tf.reshape(points, [-1, 2]))
            data = tf.data.Dataset.zip((data, kp)).map(
                    lambda d, k: {**d, 'keypoints': k})
            data = data.map(pipeline.add_valid_mask)

        # Keep only the first elements for validation
        if split_name == 'validation':
            buffer_size = len(files['image_paths'])
            # Shuffle with fixed seed for reproducibility
            data = data.shuffle(buffer_size=buffer_size, seed=42)
            data = data.take(config['validation_size'])

        # Cache to avoid always reading from disk
        if config['cache_in_memory']:
            tf.logging.info('Caching data, fist access will take some time.')
            data = data.cache()

        # Generate the warped pair
        if config['warped_pair']['enable']:
            assert has_keypoints
            warped = data.map_parallel(lambda d: pipeline.homographic_augmentation(
                d, add_homography=True, **config['warped_pair']))
            if is_training and config['augmentation']['photometric']['enable']:
                warped = warped.map_parallel(lambda d: pipeline.photometric_augmentation(
                    d, **config['augmentation']['photometric']))
            warped = warped.map_parallel(pipeline.add_keypoint_map)
            # Merge with the original data
            data = tf.data.Dataset.zip((data, warped))
            data = data.map(lambda d, w: {**d, 'warped': w})

        # Data augmentation
        if has_keypoints and is_training:
            if config['augmentation']['photometric']['enable']:
                data = data.map_parallel(lambda d: pipeline.photometric_augmentation(
                    d, **config['augmentation']['photometric']))
            if config['augmentation']['homographic']['enable']:
                assert not config['warped_pair']['enable']  # doesn't support hom. aug.
                data = data.map_parallel(lambda d: pipeline.homographic_augmentation(
                    d, **config['augmentation']['homographic']))

        # Generate the keypoint map
        if has_keypoints:
            data = data.map_parallel(pipeline.add_keypoint_map)
        data = data.map_parallel(
            lambda d: {**d, 'image': tf.to_float(d['image']) / 255.})
        if config['warped_pair']['enable']:
            data = data.map_parallel(
                lambda d: {
                    **d, 'warped': {**d['warped'],
                                    'image': tf.to_float(d['warped']['image']) / 255.}})

        return data
