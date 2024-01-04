import os
import random
import pandas as pd
import numpy as np
import cv2
import config
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from config import OG_SIZE, CLASS_NUM


class CouplerKeypointDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        super().__init__()
        random.seed(42)
        np.random.seed(42)
        self.data = pd.read_csv(csv_file)
        self.category_names = ['center_x', 'center_y', 'left_x', 'left_y', 'top_x', 'top_y', 'right_x', 'right_y', 'buttom_x', 'buttom_y', 'confidence']
        # self.category_names = ['center_x', 'center_y', 'left_x', 'left_y', 'top_x', 'top_y', 'right_x', 'right_y', 'buttom_x', 'buttom_y']
        # self.category_names = ['left_x', 'left_y', 'top_x', 'top_y', 'right_x', 'right_y', 'buttom_x', 'buttom_y']
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.train:
            # class_visibility = self.data.iloc[index]['Class']
            # src_dir = "medimages" if class_visibility else "negimages"
            src_dir= "medimages"
            src = self.data.iloc[index]['Source'].split("\\")[1:]
            src.insert(0, '/media/angel/My Passport/THA')
            src = "".join([i+"/" for i in src])
            img_path = os.path.join(src, src_dir, self.data.iloc[index]['Image'])
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            w = self.data.iloc[index]['Width']//2
            h = self.data.iloc[index]['Height']//2
            m = OG_SIZE//2
            # color = self.data.iloc[index]['Color']
            # c = 0 if color == 'black' else 1
            labels = np.array([m, m, m-w, m, m, m-h, m+w, m, m, m+h])
            # labels = np.array([m-w, m, m, m-h, m+w, m, m, m+h])
        # else:
        #     img_path = os.path.join(self.data.iloc[index]['Source'], "tb_medimages", self.data.iloc[index]['Image'])
        #     image = cv2.imread(img_path)
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     labels = np.zeros(CLASS_NUM*2)

        labels = labels.reshape(CLASS_NUM, 2)

        if self.transform:
            augmentations = self.transform(image=image, keypoints=labels)
            image = augmentations["image"]
            labels = augmentations["keypoints"]

        # if not class_visibility:
        #     labels = np.zeros(CLASS_NUM*2)

        labels = np.array(labels).reshape(-1)
        # labels = np.append(labels, class_visibility)

        return image, labels.astype(np.float32)


if __name__ == "__main__":
    ds = CouplerKeypointDataset(csv_file=r"/media/angel/My Passport/THA/tcd_code/data/TCD_FULL_SNOWY_train.csv", train=True, transform=config.train_ext_transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    for idx, (x, y) in enumerate(loader):
        plt.imshow(x[0][0].detach().cpu().numpy())
        #plt.plot(y[0][0].detach().cpu().numpy() + y[0][2].detach().cpu().numpy()//2, y[0][1].detach().cpu().numpy()+y[0][2].detach().cpu().numpy()//2, "go")
        plt.plot(y[0][0:10:2].detach().cpu().numpy(), y[0][1:11:2].detach().cpu().numpy(), "go")
        plt.show()
