from torch.utils.data import Dataset
import os
from glob import glob
from sim import Action
import cv2
import numpy as np

class LineFollowerDataset(Dataset):
    def __init__(self, dataset_dir="dataset/", transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform 
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        self.data = list()
        for input_path in glob(os.path.join(self.dataset_dir, "**/*.png")):
            action = os.path.basename(os.path.dirname(input_path))
            try:
                target = Action[action]
            except Exception:
                print(f"Invalid action {action} for input_path {input_path}")
                continue
            self.data.append({"input_path": input_path, "target": target.value})
        self.print_statistics()
        
    def print_statistics(self):
        print("LineFollowerDataset statistics:")
        print(f"> Totat datapoints: {len(self.data)}")
    
    def get_target(action:Action):
        return action.value
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datapoint = self.data[index]
        img = cv2.imread(datapoint["input_path"])
        if self.transform is not None:
            img = self.transform(img)
        return  img, datapoint["target"] 

    def extend(self, img:np.array, action:Action):
        datapoint_dir = os.path.join(self.dataset_dir, action.name)
        if not os.path.exists(datapoint_dir):
            os.makedirs(datapoint_dir)
        input_path = os.path.join(datapoint_dir, f"{len(self)}.png")
        cv2.imwrite(input_path, img)
        self.data.append({"input_path": input_path, "target": action.value})
