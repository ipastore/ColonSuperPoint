import cv2
import numpy as np
from superpoint.datasets.colon import Colon
from utils import plot_imgs
import matplotlib.pyplot as plt
import os
import warnings
import pdb

warnings.filterwarnings("ignore", category=FutureWarning)

def draw_keypoints(img, corners, color=(0, 255, 0), radius=10, s=3):
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners).T:
        cv2.circle(img, tuple(s*np.flip(c, 0)), radius, color, thickness=-1)
    return img
def draw_overlay(img, mask, color=[0, 0, 255], op=0.5, s=3):
    mask = cv2.resize(mask.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    img[np.where(mask)] = img[np.where(mask)]*(1-op) + np.array(color)*op
def display(d):
    img = draw_keypoints(d['image'][..., 0] * 255, np.where(d['keypoint_map']), (0, 255, 0)) if add_keypoints \
            else d['image'][..., 0] * 255
    draw_overlay(img, np.logical_not(d['valid_mask']))
    return img

output_root_dir = 'warped_pairs'
output_dir = f'{output_root_dir}/baseline_33_labels_rodriguez_masked_kernel5_iterations10'

os.makedirs(output_dir, exist_ok=True)


config = {
    'labels': f'/home/student/ColonSuperPoint/ColonExperiments/experiments/outputs/baseline_33_labels_rodriguez_masked_kernel5_iterations10',
    'image_path': '/home/student/ColonSuperPoint/ColonExperiments/datasets/endomapper_train/33/images',
    'preprocessing':{
        'use_colonoscopy_preprocess': True,
        'half_resolution': False,
        'camera_mask_path': '/home/student/ColonSuperPoint/ColonExperiments/datasets/endomapper_train/33/camera_mask.png',
    },
    'truncate': 50,
    'augmentation' : {
        'photometric': {
            'enable': True,
            'primitives': [
                'random_brightness', 'random_contrast', 'additive_speckle_noise',
                'additive_gaussian_noise', 'additive_shade', 'motion_blur'],
            'params': {
                'random_brightness': {'max_abs_change': 50},
                'random_contrast': {'strength_range': [0.5, 1.5]},
                'additive_gaussian_noise': {'stddev_range': [0, 10]},
                'additive_speckle_noise': {'prob_range': [0, 0.0035]},
                'additive_shade': {'transparency_range': [-.5, .5], 'kernel_size_range': [100, 150]},
                'motion_blur': {'max_kernel_size': 3},
            }
        },
    },
    'warped_pair': {
        'enable': True,
        'params': {
            'translation': True,
            'rotation': True,
            'scaling': True,
            'perspective': True,
            'scaling_amplitude': 0.2,
            'perspective_amplitude_x': 0.2,
            'perspective_amplitude_y': 0.2,
            'patch_ratio': 0.85,
            'max_angle': 1.57,
            'allow_artifacts': True,
        },
        'valid_border_margin': 3,
    },
}

if os.getenv("DEBUG") == 1:
    pdb.set_trace()


dataset = Colon(**config)
data = dataset.get_training_set()
add_keypoints = True


for i in range(5):
    d = next(data)
    plot_imgs([display(d)/255., display(d['warped'])/255.], ylabel=d['name'], 
              titles=['original', 'warped'], dpi=200, cmap='gray')
    plt.savefig(f'{output_dir}/colon_pair_{i:02d}.png', dpi=200)
    print(f'Pair {i} saved at {output_dir}/colon_pair_{i:02d}.png')
    plt.close()
