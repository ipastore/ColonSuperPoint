import tensorflow as tf
import cv2 as cv
import numpy as np

from superpoint.datasets.utils import photometric_augmentation as photaug
from superpoint.models.homographies import (sample_homography, compute_valid_mask,
                                            warp_points, filter_points, compute_extra_mask)


def parse_primitives(names, all_primitives):
    p = all_primitives if (names == 'all') \
            else (names if isinstance(names, list) else [names])
    assert set(p) <= set(all_primitives)
    return p


def photometric_augmentation(data, **config):
    with tf.name_scope('photometric_augmentation'):
        primitives = parse_primitives(config['primitives'], photaug.augmentations)
        prim_configs = [config['params'].get(
                             p, {}) for p in primitives]

        indices = tf.range(len(primitives))
        if config['random_order']:
            indices = tf.random_shuffle(indices)

        def step(i, image):
            fn_pairs = [(tf.equal(indices[i], j),
                         lambda p=p, c=c: getattr(photaug, p)(image, **c))
                        for j, (p, c) in enumerate(zip(primitives, prim_configs))]
            image = tf.case(fn_pairs)
            return i + 1, image

        _, image = tf.while_loop(lambda i, image: tf.less(i, len(primitives)),
                                 step, [0, data['image']], parallel_iterations=1)

    return {**data, 'image': image}


def homographic_augmentation(data, add_homography=False, **config):
    with tf.name_scope('homographic_augmentation'):
        image_shape = tf.shape(data['image'])[:2]
        homography = sample_homography(image_shape, **config['params'])[0]
        warped_image = tf.contrib.image.transform(
                data['image'], homography, interpolation='BILINEAR')
        
        mask = data.get('mask', None)
        if mask is not None:
            valid_mask = compute_extra_mask(mask, homography)
        else:
            valid_mask = compute_valid_mask(image_shape, homography,
                                            config['valid_border_margin'])

        warped_points = warp_points(data['keypoints'], homography)
        warped_points = filter_points(warped_points, image_shape)

    ret = {**data, 'image': warped_image, 'keypoints': warped_points,
           'valid_mask': valid_mask}
    if add_homography:
        ret['homography'] = homography
    return ret


def add_valid_mask(data):
    with tf.name_scope('valid_mask'):
        # Dummy mask for images without a mask
        valid_mask = tf.ones(tf.shape(data['image'])[:2], dtype=tf.int32)

        # Mask if available
        mask = data.get('mask', None)
        if mask is not None:
            mask = tf.squeeze(mask, axis=-1)  # remove C dim
            valid_mask = tf.where(mask, valid_mask, tf.zeros_like(valid_mask))

    return {**data, 'valid_mask': valid_mask}


def add_keypoint_map(data):
    with tf.name_scope('add_keypoint_map'):
        image_shape = tf.shape(data['image'])[:2]
        kp = tf.minimum(tf.to_int32(tf.round(data['keypoints'])), image_shape-1)
        kmap = tf.scatter_nd(
                kp, tf.ones([tf.shape(kp)[0]], dtype=tf.int32), image_shape)        
    return {**data, 'keypoint_map': kmap}


def downsample(image, coordinates, **config):
    with tf.name_scope('gaussian_blur'):
        k_size = config['blur_size']
        kernel = cv.getGaussianKernel(k_size, 0)[:, 0]
        kernel = np.outer(kernel, kernel).astype(np.float32)
        kernel = tf.reshape(tf.convert_to_tensor(kernel), [k_size]*2+[1, 1])
        pad_size = int(k_size/2)
        image = tf.pad(image, [[pad_size]*2, [pad_size]*2, [0, 0]], 'REFLECT')
        image = tf.expand_dims(image, axis=0)  # add batch dim
        image = tf.nn.depthwise_conv2d(image, kernel, [1, 1, 1, 1], 'VALID')[0]

    with tf.name_scope('downsample'):
        ratio = tf.divide(tf.convert_to_tensor(config['resize']), tf.shape(image)[0:2])
        coordinates = coordinates * tf.cast(ratio, tf.float32)
        image = tf.image.resize_images(image, config['resize'],
                                       method=tf.image.ResizeMethod.BILINEAR)

    return image, coordinates


def ratio_preserving_resize(image, **config):
    target_size = tf.convert_to_tensor(config['resize'])
    scales = tf.to_float(tf.divide(target_size, tf.shape(image)[:2]))
    new_size = tf.to_float(tf.shape(image)[:2]) * tf.reduce_max(scales)
    image = tf.image.resize_images(image, tf.to_int32(new_size),
                                   method=tf.image.ResizeMethod.BILINEAR)
    return tf.image.resize_image_with_crop_or_pad(image, target_size[0], target_size[1])

def colonoscopy_preprocess(image, **config):
    """Process colonoscopy images by cropping borders and standardizing dimensions.
    
    Args:
        image: Input image tensor [H, W, C]
        config: Configuration dictionary containing preprocessing parameters
        
    Returns:
        Processed image tensor
    """
    # Center crop to standardized dimensions (994, 1344)
    target_height = 992
    target_width = 1344

    # Calculate center crop offsets
    image_shape = tf.shape(image)
    image_height, image_width = image_shape[0], image_shape[1]
    offset_height = (image_height - target_height) // 2
    offset_width = (image_width - target_width) // 2

 
    # Apply center crop
    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, 
        target_height, 
        target_width)

    # Optionally resize to half size
    if config.get('half_resolution', False):
        new_height = target_height // 2
        new_width = target_width // 2
        image = tf.image.resize_images(
            image, [new_height, new_width], 
            method=tf.image.ResizeMethod.BILINEAR)
    
    return image