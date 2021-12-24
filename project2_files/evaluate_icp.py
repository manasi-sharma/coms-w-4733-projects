import argparse
import os

import numpy as np

from icp import obj_mesh2pts

parser = argparse.ArgumentParser()
parser.add_argument('--gtmask', action='store_true', help='evaluate predicted pose using ground truth mask')
parser.add_argument('--predmask', action='store_true', help='evaluate predicted pose using predicted mask')

LIST_OBJ_FOLDERNAME = [
        "004_sugar_box",
        "005_tomato_soup_can",
        "007_tuna_fish_can",
        "011_banana",
        "024_bowl",
    ]


def closest_point_distance(pred_pt, gt_pts):
    """
    In:
        pred_pt: Numpy array [3,], a point from point cloud sampled from mesh with predicted pose.
        gt_pts: Numpy array [N, 3], point cloud sampled from mesh with ground truth pose.
    Out:
        float, distance between pred_pt and its closest point in gt_pts.
    Purpose:
        Compute closest point distance.
    """
    distance = np.sum((gt_pts - pred_pt) ** 2, axis=1)
    return np.min(distance)


def evaluate(obj_id, pred_pose, gt_pose):
    """
    Compute average closest point distance of a specific object.
    """
    pred_pts = obj_mesh2pts(obj_id, point_num=1000, transform=pred_pose)
    gt_pts = obj_mesh2pts(obj_id, point_num=1000, transform=gt_pose)
    average_closest_point_distance = np.apply_along_axis(closest_point_distance, 1, pred_pts, gt_pts).mean()
    return average_closest_point_distance


def main():
    dataset_dir = "./dataset/val/"
    gt_pose_dir = dataset_dir + "gt_pose/"
    args = parser.parse_args()
    list_pose_dir = []
    if args.gtmask:
        list_pose_dir.append(dataset_dir + "pred_pose/gtmask/")
    if args.predmask:
        list_pose_dir.append(dataset_dir + "pred_pose/predmask/")
    if not (args.gtmask or args.predmask):
        print("Missing argument --gtmask or --predmask")
        return

    for pose_dir in list_pose_dir:
        if pose_dir == dataset_dir + "pred_pose/gtmask/":
            print("\nEvaluating predicted pose using ground truth mask")
        else:
            print("\nEvaluating predicted pose using predicted mask")
        average_closest_point = [[], [], [], [], []]
        for scene_id in range(5):
            for obj_id in range(1, 6):
                pose_filename = str(scene_id) + "_" + str(obj_id) + ".npy"
                pred_pose_name = pose_dir + pose_filename
                if not os.path.exists(pred_pose_name):
                    print(f"{pose_filename} doesn't exist")
                    continue
                pred_pose = np.load(pred_pose_name)
                gt_pose = np.load(gt_pose_dir + pose_filename)
                average_closest_point[obj_id - 1].append(evaluate(obj_id, pred_pose, gt_pose))

        for obj_id in range(1, 6):
            obj_average_closest_point = average_closest_point[obj_id - 1]
            print("Average closest point distance of", LIST_OBJ_FOLDERNAME[obj_id - 1][4:])
            print("***average:", sum(obj_average_closest_point) / len(obj_average_closest_point))
            print("min:", min(obj_average_closest_point), "max:", max(obj_average_closest_point))


if __name__ == '__main__':
    main()
