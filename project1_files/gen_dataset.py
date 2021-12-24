import os

import pybullet as p
import pybullet_data

import objects
from camera import Camera, save_obs


def main():
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)

    # Setup camera
    my_camera = Camera(
        image_size=(240, 320),
        near=0.01,
        far=10.0,
        fov_w=69.40
    )

    # For each scene, orientations of objects are guaranteed to be different, and camera distance and pitch can vary
    training_scene = 30

    # Number of observations to be made in each scene with the camera moving round a circle above the origin
    num_observation = 10

    # Load floor with ID 0
    plane_id = p.loadURDF("plane.urdf")

    # Load objects with ID start from 1
    list_obj_foldername = [
        "004_sugar_box",
        "005_tomato_soup_can",
        "007_tuna_fish_can",
        "011_banana",
        "024_bowl",
    ]
    num_obj = len(list_obj_foldername)
    list_obj_position = [[-0.1, -0.1, 0.1], [-0.1, 0.1, 0.1], [0.1, -0.1, 0.1], [0.1, 0.1, 0.1], [0, 0, 0.1], ]
    list_obj_orientation = objects.gen_obj_orientation(
        num_scene=training_scene,
        num_obj=num_obj
    )
    list_obj_id = objects.load_obj(
        list_obj_foldername,
        list_obj_position,
        list_obj_orientation
    )

    # Generate training set
    dataset_dir = "./dataset/train/"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        os.makedirs(dataset_dir + "rgb/")
        os.makedirs(dataset_dir + "gt/")
    print("Start generating the training set.")
    print(f'==> 1 / {training_scene}')
    save_obs(
        dataset_dir,
        my_camera,
        num_obs=num_observation,
        scene_id=0
    )
    for i in range(1, training_scene):
        print(f'==> {i+1} / {training_scene}')
        objects.reset_obj(
            list_obj_id,
            list_obj_position,
            list_obj_orientation,
            scene_id=i
        )
        save_obs(
            dataset_dir,
            my_camera,
            num_obs=num_observation,
            scene_id=i
        )

    p.disconnect()


if __name__ == '__main__':
    main()
