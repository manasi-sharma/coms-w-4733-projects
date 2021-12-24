import pybullet as p
import os
from enum import Enum, auto
import transformations
import numpy as np
from time import sleep
import cv2
import math # I added this


class Action(Enum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3


class WallESim:
    def __init__(self, map_path, load_landmarks=False):
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        # Load robot
        self.robot_body_id = p.loadURDF("assets/walle/walle.urdf", globalScaling=0.8)
        # Load map
        map_urdf_path = os.path.join(map_path, "map.urdf")
        if not os.path.exists(map_urdf_path):
            raise ValueError(f"Map {map_urdf_path} does not exist.")
        self.plane_id = p.loadURDF(map_urdf_path)
        # Add a marker for robot position
        robot_position, _ = p.getBasePositionAndOrientation(self.robot_body_id)
        self.robot_marker = SphereMarker(position=robot_position, rgba_color=[0, 1, 0, 0.8])
        fov, aspect, nearplane, farplane = 45.0, 1.0, 0.1, 3.1
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, nearplane, farplane)

        # load landmarks if available
        self.landmarks = np.empty((0, 3))
        self.landmark_markers = list()
        if load_landmarks:
            landmark_points_path = os.path.join(map_path, "landmarks.npy")
            if os.path.exists(landmark_points_path):
                self.landmarks = np.load(landmark_points_path)
                for landmark in self.landmarks:
                    self.landmark_markers.append(SphereMarker(landmark))
            else:
                print(f"Landmarks not found at path: {landmark_points_path}")


    def get_robot_view(self):
        position, orientation = p.getBasePositionAndOrientation(self.robot_body_id)
        camera_height = 1
        camera_lookat_distance = 1

        cameraEyePosition = [position[0],
                             position[1]-0.1, position[2]+camera_height]
        yaw = p.getEulerFromQuaternion(orientation)[2]
        yaw = yaw - np.pi / 2
        targetPosition = [
            position[0] + camera_lookat_distance * np.cos(yaw),
            position[1] + camera_lookat_distance * np.sin(yaw),
            0
        ]
        up_vector = [0, 0, 1]
        view_matrix = p.computeViewMatrix(
            cameraEyePosition, targetPosition, up_vector)
        _, _, rgbImg, _, _ = p.getCameraImage(
            224, 224, view_matrix, self.projection_matrix, flags=p.ER_NO_SEGMENTATION_MASK)
        rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2BGR)
        return rgbImg

    def move_robot(self, action: Action):
        # Action.FORWARD or Action.BACKWARD should move the robot front or back by pos_delta amount
        pos_delta = 0.05
        # Action.RIGHT and Action.LEFT should rotate the robot clockwise or anti-clockwise by rot_detal amount
        rot_delta = np.pi / 100
        # Initial robot pose in world coordinates
        pos_0, ori_0 = p.getBasePositionAndOrientation(self.robot_body_id)
        # This should contain final robot pose after performing the action in world coordinates
        pos_n, ori_n = pos_0, ori_0
        # TODO: Implement the robot movements given action
        
        # =================================================
        if(action.value==0): #FORWRD
            # specifying parameters
            advancement= -pos_delta #pos_delta
            angle= 0

        elif(action.value==1): #BACKWARD
            # specifying parameters
            advancement= pos_delta #-pos_delta
            angle= 0

        elif(action.value==2): #LEFT
            # specifying parameters
            advancement= 0
            angle= rot_delta

        elif(action.value==3): #RIGHT
            # specifying parameters
            advancement= 0
            angle= -rot_delta

        else:
            print("ERROR!")

        # new pos and ori in world coordinates
        posA= pos_0
        oriA= ori_0
        posB= np.array([0, advancement, 0])
        #oriB= p.getQuaternionFromEuler([angle, 0, 0])
        oriB= p.getQuaternionFromEuler([0, 0, angle])
        pos_n, ori_n= p.multiplyTransforms(posA, oriA, posB, oriB)
        #ori_n= p.getQuaternionFromEuler(tmp_ori_n)

        # =================================================

        p.resetBasePositionAndOrientation(self.robot_body_id, pos_n, ori_n)
        robot_position, _ = p.getBasePositionAndOrientation(self.robot_body_id)
        self.robot_marker = SphereMarker(position=robot_position, rgba_color=[0, 1, 0, 0.8])

    def step_simulation(self, num_steps=1):
        for _ in range(int(num_steps)):
            p.step_simulation()
            sleep(1e-3)

    def set_landmarks_visibility(self, is_visible):
        for landmark_marker in self.landmark_markers:
            landmark_marker.set_visibility(is_visible)


class SphereMarker:
    def __init__(self, position, radius=0.05, rgba_color=(1, 0, 0, 0.8), p_id=0):
        self.p_id = p_id
        self.rgba_color = rgba_color
        vs_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color, physicsClientId=self.p_id)

        self.marker_id = p.createMultiBody(
            baseMass=0,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vs_id,
            basePosition=np.array(position),
            useMaximalCoordinates=False
        )

    def set_visibility(self, is_visible):
        if is_visible:
            p.changeVisualShape(self.marker_id, -1, rgbaColor=self.rgba_color)
        else:
            p.changeVisualShape(self.marker_id, -1, rgbaColor=[0, 0, 0, 0])

    def __del__(self):
        p.removeBody(self.marker_id, physicsClientId=self.p_id)


def getActionFromKeyboardEvent(keys):
    if len(keys.items()) == 0:
        return False, None
    key, event_type = next(iter(keys.items()))
    if not (event_type & p.KEY_WAS_TRIGGERED) and not (event_type & p.KEY_IS_DOWN):
        return False, None
    if key == p.B3G_RIGHT_ARROW:
        action = Action.RIGHT
    elif key == p.B3G_LEFT_ARROW:
        action = Action.LEFT
    elif key == p.B3G_UP_ARROW:
        action = Action.FORWARD
    elif key == p.B3G_DOWN_ARROW:
        action = Action.BACKWARD
    else:
        return False, None
    return True, action


if __name__ == "__main__":
    map_path = "maps/train/map1"
    env = WallESim(map_path)
    while True:
        img_robot_view = env.get_robot_view()
        keys = p.getKeyboardEvents()
        is_action_valid, action = getActionFromKeyboardEvent(keys)
        if not is_action_valid:
            continue
        env.move_robot(action)
        sleep(1e-3)
