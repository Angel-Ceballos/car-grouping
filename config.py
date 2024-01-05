import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
NUM_EPOCHS = 25
NUM_WORKERS = 8
# CHECKPOINT_FILE = "best_model_ResNet18_TCD_128x128.pth"
CHECKPOINT_FILE = "placeholder.pth"
# FILE_NAME = "ResNet18_TCD_5KP_Climate_128x128x3"
FILE_NAME = "Resnet18_car"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True
OG_SIZE = 256
IMAGE_SIZE = 128
CLASS_NUM = 5

# Data augmentation for images
train_transforms = A.Compose(
    [
        A.Rotate(limit=20, p=0.25),
        A.IAAAffine(shear=15, scale=1.0, mode="constant", p=0.2),
        A.RandomBrightnessContrast(contrast_limit=0.5, brightness_limit=0.5, p=0.25),
        A.OneOf([
            A.GaussNoise(p=0.8),
            A.ImageCompression(p=0.8),
            A.RandomGamma(p=0.8),
            A.Blur(p=0.8),
        ], p=1.0),
        A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=0, p=0.5),
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.1, p=0.3),
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


test_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)