import random
import numpy
import glob
import os
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

random.seed(1301)
numpy.random.seed(1301)


def get_pred_patient_img_dir(patient_id):
    res = get_pred_patient_dir(patient_id) + "all_images\\"
    create_dir_if_not_exists(res)
    return res


def get_pred_patient_overlay_dir(patient_id):
    res = get_pred_patient_dir(patient_id) + "predicted_overlays\\"
    create_dir_if_not_exists(res)
    return res


def get_pred_patient_transparent_overlay_dir(patient_id):
    res = get_pred_patient_dir(patient_id) + "predicted_overlays_transparent\\"
    create_dir_if_not_exists(res)
    return res


def get_patient_images(patient_id):
    return get_patient_files(patient_id, "images")


def get_patient_overlays(patient_id):
    return get_patient_files(patient_id, "overlays")


def get_patient_transparent_overlays(patient_id):
    return get_patient_files(patient_id, "transparent_overlays")


def get_patient_files(patient_id, file_type, extension = ".png"):
    src_dir = get_pred_patient_dir(patient_id)
    if file_type == "images":
        src_dir = get_pred_patient_img_dir(patient_id)
    if file_type == "overlays":
        src_dir = get_pred_patient_overlay_dir(patient_id)
    if file_type == "transparent_overlays":
        src_dir = get_pred_patient_transparent_overlay_dir(patient_id)
    prefix = str(patient_id).rjust(4, '0')
    file_paths = get_files(src_dir, prefix + "*" + extension)
    return file_paths


def delete_files(target_dir, search_pattern):
    files = glob.glob(target_dir + search_pattern)
    for f in files:
        os.remove(f)


def get_files(scan_dir, search_pattern):
    file_paths = glob.glob(scan_dir + search_pattern)
    return file_paths


def compute_mean_image(src_dir, wildcard, img_size):
    mean_image = numpy.zeros((img_size, img_size), numpy.float32)
    src_files = glob.glob(src_dir + wildcard)
    random.shuffle(src_files)
    img_count = 0
    for src_file in src_files:
        if "_o.png" in src_file:
            continue
        mat = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
        mean_image += mat
        img_count += 1
        if img_count > 2000:
            break

    res = mean_image / float(img_count)
    return res


def compute_mean_pixel_values_dir(src_dir, wildcard, channels):
    src_files = glob.glob(src_dir + wildcard)
    random.shuffle(src_files)
    means = []
    for src_file in src_files:
        mat = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
        mean = mat.mean()
        means.append(mean)
        if len(means) > 10000:
            break

    res = sum(means) / len(means)
    print res
    return res


def replace_color(src_image, from_color, to_color):
    data = numpy.array(src_image)   # "data" is a height x width x 4 numpy array
    r1, g1, b1 = from_color  # Original value
    r2, g2, b2 = to_color  # Value that we want to replace it with

    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :3][mask] = [r2, g2, b2]

    return data


ELASTIC_INDICES = None  # needed to make it faster to fix elastic deformation per epoch.

def elastic_transform2d(image, alpha, sigma, random_state=None):
    global ELASTIC_INDICES
    shape = image.shape
    if ELASTIC_INDICES == None:
        if random_state is None:
            random_state = numpy.random.RandomState(1301)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
        ELASTIC_INDICES = numpy.reshape(y + dy, (-1, 1)), numpy.reshape(x + dx, (-1, 1))
    return map_coordinates(image, ELASTIC_INDICES, order=1).reshape(shape)

def elastic_transform3d(image, alpha, sigma, random_state=None):
    import numpy as np
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(1301)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


def prepare_overlay_image(src_overlay_path, target_size, antialias=False):
    if os.path.exists(src_overlay_path):
        overlay = cv2.imread(src_overlay_path)
        overlay = replace_color(overlay, (255, 255, 255), (0, 0, 0))
        overlay = replace_color(overlay, (0, 255, 255), (255, 255, 255))
        overlay = overlay.swapaxes(0, 2)
        overlay = overlay.swapaxes(1, 2)
        overlay = overlay[0]
        # overlay = overlay.reshape((overlay.shape[1], overlay.shape[2])
        interpolation = cv2.INTER_AREA if antialias else cv2.INTER_NEAREST
        overlay = cv2.resize(overlay, (target_size, target_size), interpolation=interpolation)
    else:
        overlay = numpy.zeros((target_size, target_size), dtype=numpy.uint8)
    return overlay


def create_dir_if_not_exists(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
