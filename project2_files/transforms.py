import numpy as np


def transform_is_valid(t, tolerance=1e-3):
    """
    In:
        t: Numpy array [4, 4] that is an transform candidate.
        tolerance: maximum absolute difference for two numbers to be considered close enough to each other.
    Out:
        bool: True if array is a valid transform else False.
    Purpose:
        Check if array is a valid transform.
    """
    if t.shape != (4, 4):
        return False

    rtr = np.matmul(t[:3, :3].T, t[:3, :3])
    rrt = np.matmul(t[:3, :3], t[:3, :3].T)

    inverse_check = np.isclose(np.eye(3), rtr, atol=tolerance).all() and np.isclose(np.eye(3), rrt, atol=tolerance).all()
    det_check = np.isclose(np.linalg.det(t[:3, :3]), 1.0, atol=tolerance).all()
    last_row_check = np.isclose(t[3, :3], np.zeros((1, 3)), atol=tolerance).all() and np.isclose(t[3, 3], 1.0, atol=tolerance).all()

    return inverse_check and det_check and last_row_check


def transform_concat(t1, t2):
    """
    In:
        t1: Numpy array [4, 4], left transform.
        t2: Numpy array [4, 4], right transform.
    Out:
        t1 * t2 as a numpy arrays [4x4].
    Purpose:
        Concatenate transforms.
    """
    if not transform_is_valid(t1):
        raise ValueError('Invalid input transform t1')
    if not transform_is_valid(t2):
        raise ValueError('Invalid input transform t2')

    return np.matmul(t1, t2)


def transform_point3s(t, ps):
    """
    In:
        t: Numpy array [4, 4] to represent a transform
        ps: point3s represented as a numpy array [Nx3], where each row is a point.
    Out:
        Transformed point3s as a numpy array [Nx3].
    Purpose:
        Transfrom point from one space to another.
    """
    if not transform_is_valid(t):
        raise ValueError('Invalid input transform t')

    if len(ps.shape) != 2 or ps.shape[1] != 3:
        raise ValueError('Invalid input points p')

    # convert to homogeneous
    ps_homogeneous = np.hstack([ps, np.ones((len(ps), 1), dtype=np.float32)])
    ps_transformed = np.dot(t, ps_homogeneous.T).T

    return ps_transformed[:, :3]


def transform_inverse(t):
    """
    In:
        t: Numpy array [4, 4] to represent a transform.
    Out:
        The inverse of the transform.
    Purpose:
        Find the inverse of the transfom.
    """
    if not transform_is_valid(t):
        raise ValueError('Invalid input transform t')

    return np.linalg.inv(t)


def depth_to_point_cloud(intrinsics, depth_image):
    """
    In:
        intrinsics: Numpy array [3, 3] given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]].
        depth_image: Numpy array [h, w] where each value is the z-depth value.
    Out:
        point_cloud: Numpy array [n, 3] where each row represents a different valid 3D point.
    Purpose:
        Back project a depth image to a point cloud.
        Note: point clouds are unordered, so any permutation of points in the list is acceptable.
        Note: Only output those points whose depth != 0.
    """
    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fu = intrinsics[0, 0]
    fv = intrinsics[1, 1]

    point_count = 0
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            if depth_image[v, u] > 0:
                point_count += 1

    point_cloud = np.zeros((point_count, 3))
    point_count = 0
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            if depth_image[v, u] > 0:
                point_cloud[point_count] = np.array([
                    (u - u0) * depth_image[v, u] / fu,
                    (v - v0) * depth_image[v, u] / fv,
                    depth_image[v, u]])
                point_count += 1

    return point_cloud
