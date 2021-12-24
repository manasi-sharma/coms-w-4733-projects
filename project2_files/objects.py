import numpy as np
import pybullet as p


def gen_obj_orientation(num_scene, num_obj):
    """
    In:
        num_scene: int, number of scenes.
        num_obj: int, number of objects.
    Out:
        list_obj_orientation: a list of 3 floats,
                              accumulating 3 rotations in radians, expressing the X,Y,Z Euler angles:
                              the roll around the X, pitch around Y and yaw around the Z axis.
    Purpose:
        Randomly generate a list of orientation for each object in each scene, without duplication.
    """
    list_obj_orientation = []
    num_ori = num_scene * num_obj
    np.random.seed(42)
    list_roll = np.random.choice(360, num_ori, replace=False)
    list_pitch = np.random.choice(360, num_ori, replace=False)
    list_yaw = np.random.choice(360, num_ori, replace=False)

    # degree to radius
    list_roll = [x/180*np.pi for x in list_roll]
    list_pitch = [x / 180 * np.pi for x in list_pitch]
    list_yaw = [x / 180 * np.pi for x in list_yaw]

    for i in range(num_ori):
        list_obj_orientation.append(
            [list_roll[i],
             list_pitch[i],
             list_yaw[i]]
        )
    return list_obj_orientation


def load_obj(name, position, orientation):
    """Load objects."""
    list_obj_id = []
    num_obj = len(name)
    for i in range(num_obj):
        cur_orientation = orientation[i]
        cur_id = p.loadURDF(
            fileName="./YCB_subsubset/" + name[i] + "/obj.urdf",
            basePosition=position[i],
            baseOrientation=p.getQuaternionFromEuler(
                [cur_orientation[0],
                 cur_orientation[1],
                 cur_orientation[2]]
            ),
            globalScaling=1,
        )
        list_obj_id.append(cur_id)
    # Drop objects on the floor
    for tick in range(500):
        p.stepSimulation()
    return list_obj_id


def reset_obj(list_obj_id, position, orientation, scene_id):
    """Reset objects."""
    num_obj = len(list_obj_id)
    np.random.seed(scene_id)
    position_index = np.random.choice(5, 5, replace=False)
    for i in range(num_obj):
        cur_orientation = orientation[scene_id * num_obj + i]
        p.resetBasePositionAndOrientation(
            list_obj_id[i],
            posObj=position[position_index[i]],
            ornObj=p.getQuaternionFromEuler(
                [cur_orientation[0],
                 cur_orientation[1],
                 cur_orientation[2]]
            )
        )
    # Drop objects on the floor
    for tick in range(500):
        p.stepSimulation()
    return
