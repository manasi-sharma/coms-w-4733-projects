import math

import numpy as np
import pybullet as p

import image


class Camera(object):
    """
    Class to define a camera.
    Modified from Zhenjia Xu's camera setting.
    """
    def __init__(self, image_size, near, far, fov_w):
        """
        In:
            image_size: tuple of (height, width), where the height and width are integer.
            near: float, value of near plane.
            far: float, value of far plane.
            fov_w: float, field of view in width direction in degree.
        Out:
            None.
        Purpose:
            Create a camera from given parameters.
        """
        super().__init__()

        # Set from input args
        self.image_size = image_size
        self.near = near
        self.far = far
        self.fov_width = fov_w
        self.focal_length = (float(self.image_size[1]) / 2) / np.tan((np.pi * self.fov_width / 180) / 2)
        self.fov_height = (math.atan((float(self.image_size[0]) / 2) / self.focal_length) * 2 / np.pi) * 180
        self.intrinsic_matrix, self.projection_matrix = self.compute_camera_matrix()

    def compute_camera_matrix(self):
        """
        In:
            None.
        Out:
            intrinsic_matrix: Numpy array [3, 3].
            projection_matrix: a list of 16 floats, representing a 4x4 matrix.
        Purpose:
            Compute intrinsic and projection matrices from instance variables in Camera class.
        """
        intrinsic_matrix = np.eye(3)  # TODO: compute camera intrinsic_matrix from camera parameters
        # assuming that the origin is at the right-bottom corner
        u0= self.image_size[1]/2
        v0= self.image_size[0]/2

        intrinsic_matrix[0][0]= self.focal_length
        intrinsic_matrix[1][1]= self.focal_length
        intrinsic_matrix[0][2]= u0
        intrinsic_matrix[1][2]= v0

        aspect_ratio= float(self.image_size[1])/float(self.image_size[0])
        projection_matrix = p.computeProjectionMatrixFOV(fov=self.fov_height, aspect=aspect_ratio, nearVal=self.near, farVal=self.far)  # TODO: compute projection_matrix from camera parameters
        return intrinsic_matrix, projection_matrix


def cam_view2pose(cam_view_matrix):
    """
    In:
        cam_view_matrix: a list of 16 floats, representing a 4x4 matrix.
    Out:
        cam_pose_matrix: Numpy array [4, 4].
    Purpose:
        Convert camera view matrix to pose matrix.
    """
    cam_pose_matrix = np.linalg.inv(np.array(cam_view_matrix).reshape(4, 4).T)
    cam_pose_matrix[:, 1:3] = -cam_pose_matrix[:, 1:3]
    return cam_pose_matrix


def make_obs(camera, view_matrix):
    """
    In:
        camera: Camera instance.
        view_matrix: a list of 16 floats, representing a 4x4 matrix.
    Out:
        rgb_obs: Numpy array [Height, Width, 3].
        depth_obs: Numpy array [Height, Width].
        mask_obs: Numpy array [Height, Width].
    Purpose:
        Use a camera to make observation and return RGB, depth and instance level segmentation mask observations.
    """
    obs = p.getCameraImage(
        width=camera.image_size[1],
        height=camera.image_size[0],
        viewMatrix=view_matrix,
        projectionMatrix=camera.projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        shadow=1,
    )

    need_convert = False
    if type(obs[2]) is tuple:
        need_convert = True

    if need_convert:
        rgb_pixels = np.asarray(obs[2]).reshape(camera.image_size[0], camera.image_size[1], 4)
        rgb_obs = rgb_pixels[:, :, :3]
        z_buffer = np.asarray(obs[3]).reshape(camera.image_size[0], camera.image_size[1])
        depth_obs = camera.far * camera.near / (camera.far - (camera.far - camera.near) * z_buffer)
        mask_obs = np.asarray(obs[4]).reshape(camera.image_size[0], camera.image_size[1])
    else:
        rgb_obs = obs[2][:, :, :3]
        depth_obs = camera.far * camera.near / (camera.far - (camera.far - camera.near) * obs[3])
        mask_obs = obs[4]

    mask_obs[mask_obs == -1] = 0  # label empty space as 0, the same as the plane
    return rgb_obs, depth_obs, mask_obs


def save_obs(dataset_dir, camera, num_obs, scene_id, is_valset=False):
    """
    In:
        dataset_dir: string, the directory to save observations.
        camera: Camera instance.
        num_obs: int, number of observations to be made in current scene with the camera moving round a circle above the origin.
        scene_id: int, indicating the scene to observe, used to (1) index files to be saved (2) change camera distance and pitch angle.
    Out:
        None.
    Purpose:
        Save RGB, depth, instance level segmentation mask as files.
    """
    for i in range(num_obs):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.0, 0.0, 0.0),
            distance=0.7 - scene_id * 0.005,
            yaw=i * (360 / num_obs),  # move round a circle orbit
            pitch=-20 - scene_id * 0.5,
            roll=0,
            upAxisIndex=2,
        )
        rgb_obs, depth_obs, mask_obs = make_obs(camera, view_matrix)
        rgb_name = dataset_dir + "rgb/" + str(i + scene_id * num_obs) + "_rgb.png"
        image.write_rgb(rgb_obs.astype(np.uint8), rgb_name)
        mask_name = dataset_dir + "gt/" + str(i + scene_id * num_obs) + "_gt.png"
        image.write_mask(mask_obs, mask_name)
