import torch
import numpy as np
import config
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import seaborn as sn
from sklearn.metrics import confusion_matrix
import itertools

plt.style.use('ggplot')


def get_prediction(loader, model, device):
    """
    Predict and return dataframe of pred vs gt
    """
    rows = []
    model.eval()
    for batch_idx, (data, targets) in enumerate(loader):
        image = data.to(device=device)
        label1, label2, label3 = targets['color'].item().__round__(), targets['type'].item().__round__(), targets['orientation'].item().__round__()
    
        # forward
        output = model(image)
        # softmax
        label1_hat=torch.argmax(output['label1']).cpu().item()
        label2_hat=torch.argmax(output['label2']).cpu().item()
        label3_hat=torch.argmax(output['label3']).cpu().item()
        rows.append({'pred_color':label1_hat, 'pred_type':label2_hat, 'pred_orientation':label3_hat,
                     'gt_color':label1, 'gt_type':label2, 'gt_orientation':label3})
 
    df = pd.DataFrame(rows)
    return df


def get_rmse(loader, model, loss_fn, device):
    model.eval()
    num_examples = 0
    losses = []
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = loss_fn(scores[targets != -1], targets[targets != -1])
        num_examples += scores[targets != -1].shape[0]
        losses.append(loss.item())

    model.train()
    loss_avg = (sum(losses)/num_examples)**0.5
    print(f"Loss on val: {loss_avg}\n")
    return loss_avg

def get_cross_loss(loader, model, loss_fn, device):
    model.eval()
    valid_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loader):
            image = data.to(device=device)
            label1, label2, label3 = targets['color'].to(device), targets['type'].to(device), targets['orientation'].to(device)
          
            output = model(image)
            label1_hat=output['label1']
            label2_hat=output['label2']
            label3_hat=output['label3']               
            # calculate loss
            loss1=loss_fn(label1_hat, label1.type(torch.LongTensor).to(device))
            loss2=loss_fn(label2_hat, label2.type(torch.LongTensor).to(device))
            loss3=loss_fn(label3_hat, label3.type(torch.LongTensor).to(device))     
            loss=loss1+loss2+loss3
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
    model.train()
    print(f"Loss on val: {valid_loss}\n")
    return valid_loss.cpu()

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def build_model(pretrained=True, fine_tune=True, num_classes=1, m_name='ResNet'):
    """
    Function to build the neural network model. Returns the final model.

    Parameters
    :param pretrained (bool): Whether to load the pre-trained weights or not.
    :param fine_tune (bool): Whether to train the hidden layers or not.
    :param num_classes (int): Number of classes in the dataset. 
    """
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    elif not pretrained:
        print('[INFO]: Not loading pre-trained weights')
    
    if m_name == 'ResNet':
        model = models.resnet18(pretrained=pretrained)
    elif m_name == 'EffNet':
        model = models.efficientnet_b0(pretrained=pretrained)
    else:
        raise Exception("Network name not recognize")

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
            
    # change the final classification head, it is trainable
    if m_name == 'ResNet':
        # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(512, num_classes)
    elif m_name == 'EffNet':
        model.classifier[1] = nn.Linear(1280, num_classes)
    return model

def export_model(input, model, name):
    tracedModel = torch.jit.trace(model, input)
    tracedModel.save(name + ".pt")

class SaveBestModel:
    """
    https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, filename, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.filename = filename
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'model/best_model_{self.filename}_L-{current_valid_loss:.2f}.pth')

def save_model(epochs, model, optimizer, criterion, filename):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'model/final_model_{filename}.pth')

def save_plots(train_loss, valid_loss, filename):
    # train_acc, valid_acc,
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    # plt.figure(figsize=(10, 7))
    # plt.plot(
    #     train_acc, color='green', linestyle='-', 
    #     label='train accuracy'
    # )
    # plt.plot(
    #     valid_acc, color='blue', linestyle='-', 
    #     label='validataion accuracy'
    # )
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.savefig('outputs/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'model/loss_{filename}.png')

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `norimport pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrixmalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')