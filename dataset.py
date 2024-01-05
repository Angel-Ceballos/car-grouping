import os
import random
import pandas as pd
import numpy as np
import cv2
import config
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from config import OG_SIZE, CLASS_NUM


class CarDataset(Dataset):
    def __init__(self, csv_file, train=True, debug=False, transform=None):
        super().__init__()
        random.seed(42)
        np.random.seed(42)
        self.data = pd.read_csv(csv_file)
        self.category_names = ['color', 'type', 'orientation']
        self.colors = {'red':0, 'green':1, 'yellow':2, 'orange':3, 'blue':4, 'purple':5, 
                       'black':6, 'white':7, 'brown':8, 'gray':9}
        self.types = {'hatchback':0, 'sedan':1, 'minivan':2, 'van':3, 'suv':4, 'pickup':5, 'truck':6, 'bus':7}
        self.orientations = {'front':0, 'back':1, 'fl':2, 'fr':3, 'bl':4, 'br':5, 'left':6, 'right':7}
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        src_dir= "/home/angel/computer_vision/car-grouping/subset"
        img_path = os.path.join(src_dir, self.data.iloc[index]['file']+'.jpg')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = {}

        if self.train:
            color = np.array(self.colors.get(self.data.iloc[index]['color'])).astype(np.float32)
            c_type = np.array(self.types.get(self.data.iloc[index]['type'])).astype(np.float32)
            orientation = np.array(self.orientations.get(self.data.iloc[index]['orientation'])).astype(np.float32)
            labels = {'color': torch.from_numpy(color),
                      'type': torch.from_numpy(c_type),
                      'orientation': torch.from_numpy(orientation)}

        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
        

        return image, labels


if __name__ == "__main__":
    ds = CarDataset(csv_file="./labels/train_data.csv", train=True, transform=config.train_transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    for idx, (x, y) in enumerate(loader):
        img = x[0][0].detach().cpu().numpy()
        plt.imshow(img)
        plt.axis("off")
        #plt.plot(y[0][0].detach().cpu().numpy() + y[0][2].detach().cpu().numpy()//2, y[0][1].detach().cpu().numpy()+y[0][2].detach().cpu().numpy()//2, "go")
        # plt.plot(y[0][0:10:2].detach().cpu().numpy(), y[0][1:11:2].detach().cpu().numpy(), "go")
        plt.show()
