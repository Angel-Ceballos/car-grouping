import torch
from dataset import CarDataset
from torch import nn, optim
import os
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import functional as F
from utils import (
    get_cross_loss,
    SaveBestModel,
    save_model,
    save_plots,
)
import time

start_time = time.time()

train_loss, valid_loss = [], []

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    t_loss = 0.0
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        image = data.to(device=device)
        label1, label2, label3 = targets['color'].to(device), targets['type'].to(device), targets['orientation'].to(device)
        # data = data.to(device=device)
        # targets = targets.to(device=device)

        # forward
        optimizer.zero_grad()
        output = model(image)
        label1_hat=output['label1']
        label2_hat=output['label2']
        label3_hat=output['label3'] 

        # calculate loss
        loss1=loss_fn(label1_hat, label1.type(torch.LongTensor).to(device))
        loss2=loss_fn(label2_hat, label2.type(torch.LongTensor).to(device))
        loss3=loss_fn(label3_hat, label3.type(torch.LongTensor).to(device))     
        loss=loss1+loss2+loss3
        # back prop
        loss.backward()
        # grad
        optimizer.step()
        t_loss = t_loss + ((1 / (batch_idx + 1)) * (loss.data - t_loss))
        # if batch_idx % 50 == 0:
        #     print('Batch %d loss: %.6f' %
        #         (batch_idx + 1, t_loss))
    
    train_loss.append(t_loss.cpu())
    print(f"\nLoss average over epoch: {t_loss}\n")

class CNN(nn.Module):
    def __init__(self, pretrained):
        super(CNN, self).__init__()
        if pretrained is True:
            self.model = resnet18(weights='DEFAULT')
        else:
            self.model = resnet18()
            
        self.fc1 = nn.Linear(512, 10)  #For color class
        self.fc2 = nn.Linear(512, 8)    #For type class
        self.fc3 = nn.Linear(512, 8)    #For orientation class
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        label1 = self.fc1(x)
        label2 = self.fc2(x)
        label3 = self.fc3(x)
        return {'label1': label1, 'label2': label2, 'label3': label3}

def main():
    train_ds = CarDataset(
        csv_file=r"./labels/train_data.csv",
        transform=config.train_transforms,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
    )
    val_ds = CarDataset(
        transform=config.val_transforms,
        csv_file=r"./labels/val_data.csv",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )

    model = CNN(pretrained=True)
    model = model.to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    # initialize SaveBestModel class
    save_best_model = SaveBestModel(config.FILE_NAME)


    for epoch in range(config.NUM_EPOCHS):
        print(f"Training Epoch:{epoch}/{config.NUM_EPOCHS}\n")
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)
        v_loss = get_cross_loss(val_loader, model, loss_fn, config.DEVICE)
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
