import pybullet as p
import argparse
from sim import Action, WallESim, getActionFromKeyboardEvent
from dataset import LineFollowerDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HW4: Data Collection")
    parser.add_argument("--map_path", "-m", type=str, default="maps/train/map1",
                        help="path of map directory. eg: maps/train/map2")
    args = parser.parse_args()

    env = WallESim(args.map_path)
    dataset = LineFollowerDataset()

    while True:
        img_robot_view = env.get_robot_view()
        keys = p.getKeyboardEvents()
        is_action_valid, action = getActionFromKeyboardEvent(keys)
        if not is_action_valid:
            continue
        env.move_robot(action)
        dataset.extend(img_robot_view, action)
