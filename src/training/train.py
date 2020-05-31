import sys
sys.path.append('..')

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from detection.models import SSD
from detection.data.dataset import USGDataset
from detection.models.utils import MultiBoxLoss, SSDBox

data_file = '../../data/sample_data.txt'
batch_size = 2
num_classes = 2
lr = 1e-3
weight_decay = 1e-4
num_epochs = 10

ssdbox = SSDBox()
def transform(img, bbox, label):
    transformation = transforms.Compose([
                transforms.Resize((300,300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),
            ])
    
    img = transformation(img)
    bbox, label = ssdbox.encode(bbox, label)

    return img, bbox, label

dataset = USGDataset(data_file, root_dir='../../data/', transform=transform)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = SSD('resnet101', num_classes=num_classes, pretrained=False)

criterion = MultiBoxLoss(num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def train(model, dataloader, criterion, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(dataloader):

        optimizer.zero_grad()
        loc_preds, cls_preds = model(inputs)
        loc_loss, conf_loss, loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        print('\rtrain_loss: %.3f | avg_loss: %.3f [%d/%d]'% (loss.item(), train_loss/(batch_idx+1), batch_idx+1, len(dataloader)), end='')
    avg_loss = train_loss/len(dataloader)
    print('\rtrain_loss: %.3f | avg_loss: %.3f [%d/%d]'% (loss.item(), avg_loss, len(dataloader), len(dataloader)))

    return avg_loss


def val(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(dataloader):

        loc_preds, cls_preds = model(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()

        val_loss += loss.item()

        print('\rval_loss: %.3f | avg_loss: %.3f [%d/%d]'% (loss.item(), val_loss/(batch_idx+1), batch_idx+1, len(dataloader)), end='')
    avg_loss = val_loss/len(dataloader)
    print('\rval_loss: %.3f | avg_loss: %.3f [%d/%d]'% (loss.item(), avg_loss, len(dataloader), len(dataloader)))

    return avg_loss


for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer)
    val_loss = val(model, val_dataloader, criterion)