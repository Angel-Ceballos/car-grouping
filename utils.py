import torch
import numpy as np
import config
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
# from torchvision.ops.boxes import _box_inter_union
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn

plt.style.use('ggplot')


def get_prediction(loader, model, device, out_name):
    """
    """
    predictions = []
    images = []
    gt_list = []
    model.eval()
    inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        norm_data = inv_normalize(data)

        # forward
        scores = model(data)
        preds = scores.cpu().detach().numpy()
        predictions.append(preds[0].tolist())
        img = norm_data[0].detach().cpu().numpy()
        img = img.transpose(1, 2, 0).tolist()
        gt = targets[0].cpu().numpy().tolist()
        images.append(img)
        gt_list.append(gt)
 
    df = pd.DataFrame({"RowId": np.arange(1, len(predictions)+1), "Prediction": predictions, "GT": gt_list, "Image": images})
    df.to_csv(out_name+".csv", index=False)
    model.train()

# def giou_loss(input_boxes, target_boxes, eps=1e-7):
#     """
#     Args:
#         input_boxes: Tensor of shape (N, 4) or (4,).
#         target_boxes: Tensor of shape (N, 4) or (4,).
#         eps (float): small number to prevent division by zero
#     """
#     inter, union = _box_inter_union(input_boxes, target_boxes)
#     iou = inter / union

#     # area of the smallest enclosing box
#     min_box = torch.min(input_boxes, target_boxes)
#     max_box = torch.max(input_boxes, target_boxes)
#     area_c = (max_box[:, 2] - min_box[:, 0]) * (max_box[:, 3] - min_box[:, 1])

#     giou = iou - ((area_c - union) / (area_c + eps))

#     loss = 1 - giou

#     return loss.sum()

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
                }, f'data/best_model_{self.filename}_L-{current_valid_loss:.2f}.pth')

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
                }, f'data/final_model_{filename}.pth')

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
    plt.savefig(f'data/loss_{filename}.png')