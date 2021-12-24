import cv2
import numpy as np


def write_grayscale(image, file_path):
    """
    In:
        image:  Grayscale image as np array [height, width], each value in range [0, 255].
        file_path: Output png file path.
    Out:
        None.
    Purpose:
        Write out a grayscale image.
    """
    cv2.imwrite(file_path, image)


def read_grayscale(file_path):
    """
    In:
        file_path: Grayscale image png to read.
    Out:
        Grayscale image as np array [height, width], each value in range [0, 255].
    Purpose:
        Read in a grayscale image.
    """
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)


def write_rgb(image, file_path):
    """
    In:
        image:  RGB image as np array [height, width, 3], each value in range [0, 255]. Color channel in the order RGB.
        file_path: Output png file path.
    Out:
        None.
    Purpose:
        Write out a color image.
    """
    cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def read_rgb(file_path):
    """
    In:
        file_path: Color image png to read.
    Out:
        RGB image as np array [height, width, 3], each value in range [0, 255]. Color channel in the order RGB.
    Purpose:
        Read in a color image.
    """
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)


def write_depth(depth_image, file_path):
    """
    In:
        depth_image: Depth data as np array [height, width], where each value is z depth in meters.
        file_path: Output png file path.
    Out:
        None.
    Purpose:
        Write a depth image (input in meters) to a 16-bit png, where depth is stored in mm.
    """
    # convert from depth in meters to millimeters
    depth_image = depth_image * 1000.

    depth_image = depth_image.astype(np.uint16)
    cv2.imwrite(file_path, depth_image)


def read_depth(file_path):
    """
    In:
        file_path: Path to depth png image saved as 16-bit z depth in mm.
    Out:
        depth_image: np array [height, width].
    Purpose:
        Read in a depth image.
    """
    # depth is saved as 16-bit uint in millimenters
    depth_image = cv2.imread(file_path, -1).astype(float)

    # millimeters to meters
    depth_image /= 1000.

    return depth_image


def write_mask(mask, file_path):
    """
    In:
        image: Segmentation mask as np array [height, width], with values correspond to object ID.
        file_path: Output png file path.
    Out:
        None.
    Purpose:
        Write out a mask image.
    """
    cv2.imwrite(file_path, mask.astype(np.uint8))


def read_mask(file_path):
    """
    In:
        file_path: Path to mask png image saved as 8-bit integer.
    Out:
        Segmentation mask as np array [height, width].
    Purpose:
        Read in a mask image.
    """
    return cv2.imread(file_path, -1)
