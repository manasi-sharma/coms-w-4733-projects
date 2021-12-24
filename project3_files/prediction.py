import argparse
from os import close
from sim import Action, WallESim
import pybullet as p
import torch
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
import cv2
import random # I added this
from train import load_chkpt # I added this
from torchvision.models import resnet18 # I added this
from torch import nn # I added this
import copy # I added this

CLOSE_DISTANCE_THRESHOLD = 0.3

class ImgProcessingActionPredictor:
    prev_action= Action.FORWARD # I added this
    prev_mid_point= 0 # I added this
    flag= False

    def __init__(self):
        pass

    def predict_action(self, img):
        action = Action.FORWARD
        # TODO:
        # ===============================================
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row_num= 200 # near the bottom of image
        hor_slice= gray_img[row_num]
        threshold_intensity= np.where(hor_slice < 50, 0, 1)
        indices_white_line= np.where(threshold_intensity == 1)

        if(indices_white_line[0].size == 0):
            return Action.RIGHT

        mid_point_white_line= round(np.average([indices_white_line[0][0], indices_white_line[0][-1]]))
        mid_point= round(len(threshold_intensity)/2.0)

        if(mid_point == mid_point_white_line):
            action= Action.FORWARD
        elif(mid_point_white_line > mid_point):
            action= Action.RIGHT
        elif(mid_point_white_line < mid_point):
            action= Action.LEFT

        # option 1
        if( (action == Action.RIGHT and self.prev_action == Action.LEFT) or (action == Action.LEFT and self.prev_action == Action.RIGHT) ):
            self.flag= True
            if( abs(mid_point_white_line - mid_point) >  abs(self.prev_mid_point - mid_point) ):
                action= self.prev_action

        # option 2
        if( self.flag ):
            action= Action.FORWARD
            self.flag= False

        # updating previous action
        self.prev_action= action
        self.prev_mid_point= mid_point_white_line

        # ===============================================
        return action


class ImitationLearningActionPredictor:
    model = None

    def __init__(self, model_path, transform=None):
        # TODO: Load model.
        # ===============================================
        self.model= copy.deepcopy(resnet18(pretrained=True))
        top_model= nn.Sequential(
                nn.Linear(self.model.fc.in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 4), # change to 2 or 1
                nn.ReLU(inplace=True))
        self.model.fc = top_model
        self.model= load_chkpt(self.model, model_path)
        # ===============================================

    def predict_action(self, img):
        action = Action.FORWARD
        # TODO:
        # ===============================================
        self.model.eval()
        #transform=transforms.Compose([
        #    transforms.ToTensor()])
        with torch.no_grad():

            transform=transforms.ToTensor()
            img_tensor= transform(img)
            img_tensor= img_tensor.unsqueeze(0)
            #img_tensor= transforms.ToTensor()(img)
            output= self.model(img_tensor)
            _, preds = torch.max(output, 1)
            pred= preds.item()

            if(pred == 0):
                action= Action.FORWARD
            elif(pred == 1):
                action= Action.BACKWARD
            elif(pred == 2):
                action= Action.LEFT
            elif(pred == 3):
                action= Action.RIGHT

        # ===============================================
        return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HW4: Testing line following algorithms")
    parser.add_argument("--use_imitation_learning", "-uip", action="store_true",
                        help="Algorithm to use: 0->image processing, 1->trained model")
    parser.add_argument("--map_path", "-m", type=str, default="maps/test/map1",
                        help="path to map directory. eg: maps/test/map2")
    parser.add_argument("--model_path", type=str, default="following_model.pth",
                        help="Path to trained imitation learning based action predictor model")
    args = parser.parse_args()

    env = WallESim(args.map_path, load_landmarks=True)

    if args.use_imitation_learning:
        # TODO: Provide transform arguments if any to the constructor
        # =================================================================
        actionPredictor = ImitationLearningActionPredictor(args.model_path)
        # =================================================================
    else:
        actionPredictor = ImgProcessingActionPredictor()

    landmarks_reached = np.zeros(len(env.landmarks), dtype=np.bool)
    assert len(landmarks_reached) != 0
    iteration = 1
    while True:
        env.set_landmarks_visibility(False)
        rgbImg = env.get_robot_view()
        env.set_landmarks_visibility(True)
        action = actionPredictor.predict_action(rgbImg)
        env.move_robot(action)

        position, _ = p.getBasePositionAndOrientation(env.robot_body_id)
        distance_from_landmarks = np.linalg.norm(env.landmarks - position, axis=1)
        closest_landmark_index = np.argmin(distance_from_landmarks)
        if distance_from_landmarks[closest_landmark_index] < CLOSE_DISTANCE_THRESHOLD and not landmarks_reached[closest_landmark_index]:
            landmarks_reached[closest_landmark_index] = True
        print(
            f"[{iteration}] {np.sum(landmarks_reached)} / {len(landmarks_reached)} landmarks reached. "
            f"{distance_from_landmarks[closest_landmark_index]:.2f} distance away from nearest landmark"
        )
        if np.all(landmarks_reached):
            print("All landmarks reached!")
            break
        iteration += 1
