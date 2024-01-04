import torch
from dataset import CouplerKeypointDataset
from torch import nn, optim
import os
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet18, resnet50
from models.resnet_pytorch import ResNet50
from models.resnext_pytorch import ResNeXt50
from models.cspresnext_pytorch import CspResNeXt50  
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_rmse,
    SaveBestModel,
    save_model,
    save_plots,
)
import time

start_time = time.time()

train_loss, valid_loss = [], []

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)
    num_examples = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        scores[targets == -1] = -1
        loss = loss_fn(scores, targets)
        num_examples += torch.numel(scores[targets != -1])
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_avg = (sum(losses)/num_examples)**0.5
    train_loss.append(loss_avg)
    print(f"\nLoss average over epoch: {loss_avg}\n")


def main():
    train_ds = CouplerKeypointDataset(
        csv_file=r"/media/angel/My Passport/THA/tcd_code/data/TCD_FULL_SNOWY_train.csv",
        transform=config.train_ext_transforms,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
    )
    val_ds = CouplerKeypointDataset(
        transform=config.val_transforms,
        csv_file=r"/media/angel/My Passport/THA/tcd_code/data/TCD_FULL_SNOWY_test.csv",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )

    # test_ds = CouplerKeypointDataset(
    #     csv_file="data/test.csv",
    #     transform=config.val_transforms,
    #     train=False,
    # )

    # test_loader = DataLoader(
    #     test_ds,
    #     batch_size=1,
    #     num_workers=config.NUM_WORKERS,
    #     pin_memory=config.PIN_MEMORY,
    #     shuffle=False,
    # )
    loss_fn = nn.MSELoss(reduction="sum")
    # model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # model.classifier[1] = nn.Linear(1280, config.CLASS_NUM*2)
    model = CspResNeXt50(num_cls=config.CLASS_NUM*2)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.fc = nn.Linear(2048, config.CLASS_NUM*2)
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()
    # initialize SaveBestModel class
    save_best_model = SaveBestModel(config.FILE_NAME)

    # model_4 = EfficientNet.from_pretrained("efficientnet-b0")
    # model_4._fc = nn.Linear(1280, 30)
    # model_15 = EfficientNet.from_pretrained("efficientnet-b0")
    # model_15._fc = nn.Linear(1280, 30)
    # model_4 = model_4.to(config.DEVICE)
    # model_15 = model_15.to(config.DEVICE)

    # if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
    #     load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)
    #     load_checkpoint(torch.load("b0_4.pth.tar"), model_4, optimizer, config.LEARNING_RATE)
    #     load_checkpoint(torch.load("b0_15.pth.tar"), model_15, optimizer, config.LEARNING_RATE)

    # get_submission(test_loader, test_ds, model_15, model_4)

    for epoch in range(config.NUM_EPOCHS):
        print(f"Training Epoch:{epoch}/{config.NUM_EPOCHS}\n")
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)
        v_loss = get_rmse(val_loader, model, loss_fn, config.DEVICE)
        valid_loss.append(v_loss)
        # save the best model till now if we have the least loss in the current epoch
        if config.SAVE_MODEL:
            save_best_model(
                v_loss, epoch, model, optimizer, loss_fn 
            )
            print('-'*50)


    # save the trained model weights for a final time
    save_model(config.NUM_EPOCHS, model, optimizer, loss_fn, config.FILE_NAME)
    # save the loss and accuracy plots
    save_plots(train_loss, valid_loss, config.FILE_NAME)
    print('TRAINING COMPLETE')

if __name__ == "__main__":
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
