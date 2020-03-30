import time
import json
import tqdm
import prettytable
import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
import albumentations
import albumentations.pytorch as AT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import torchvision.models as models
from data import LeafDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = pd.read_csv('train.csv')
class_names = list(train['labels'].value_counts().sort_index().index)
num_classes = len(class_names)

train, valid = model_selection.train_test_split(train, stratify=train.labels, test_size=0.1, random_state=2020)
train_counts = train['labels'].value_counts().sort_index().values
valid_counts = valid['labels'].value_counts().sort_index().values
print("Dataset Statistics:\n" + "===================\n")
table = prettytable.PrettyTable(['Class', 'Train', 'Valid'])
for i in range(num_classes):
    table.add_row([class_names[i], train_counts[i], valid_counts[i]])
print(table)
print()
      
train_transform = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.CenterCrop(224, 224),
    albumentations.HorizontalFlip(),
    albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
    albumentations.RandomBrightness(),
    albumentations.HueSaturationValue(),
    albumentations.Normalize(),
    AT.ToTensor()
])
valid_transform = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.CenterCrop(224, 224),
    albumentations.Normalize(),
    AT.ToTensor()
])

train_dataset = LeafDataset(df=train, image_dir='G:\\Plant Pathology 2020\\images', transform=train_transform, smooth_factor=0)
valid_dataset = LeafDataset(df=valid, image_dir='G:\\Plant Pathology 2020\\images', transform=valid_transform, smooth_factor=0)

train_loader = utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=False)

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

for p in model.parameters():
    p.requires_grad = True
params = filter(lambda p: p.requires_grad, model.parameters())

optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.001)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

since = time.time()
history = {
    'train': {'loss': [], 'acc': []},
    'valid': {'loss': [], 'acc': []},
    'lr': []
}
best_acc = 0.0

print("Start training ...\n" + "==================\n")
num_epochs = 30
for epoch in range(1, num_epochs + 1):
    head = 'epoch {:2}/{:2}'.format(epoch, num_epochs)
    print(head + '\n' + '-'*(len(head)))

    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm.tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        weights = (1.0 - 0.99) / np.array(1.0 - np.power(0.99, train_counts))
        weights = weights / np.sum(weights) * num_classes
        weights = torch.tensor(weights)
        
        weights = weights.unsqueeze(0)
        weights = weights.to(device)
        weights = weights.repeat(labels.shape[0], 1)
        weights = weights * labels
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, num_classes)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float(), weights.float())

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == torch.max(labels, 1)[1])

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    history['train']['loss'].append(epoch_loss)
    history['train']['acc'].append(epoch_acc.item())
    print('{} - loss: {:.4f} acc: {:.2f}'.format('train', epoch_loss, epoch_acc * 100))

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm.tqdm(valid_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == torch.max(labels, 1)[1])

    epoch_loss = running_loss / len(valid_dataset)
    epoch_acc = running_corrects.double() / len(valid_dataset)
    history['valid']['loss'].append(epoch_loss)
    history['valid']['acc'].append(epoch_acc.item())
    print('{} - loss: {:.4f} acc: {:.2f}'.format('valid', epoch_loss, epoch_acc * 100)) 

    history['lr'].append(optimizer.param_groups[0]['lr'])
    lr_scheduler.step(epoch_loss)

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), 'model.pth')

with open('history.json', 'w') as f:
    json.dump(history, f)

time_elapsed = time.time() - since
print('\nTraining time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))