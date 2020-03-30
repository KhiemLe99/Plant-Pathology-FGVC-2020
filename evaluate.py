import cv2
import tqdm
import json
import pandas as pd
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics
import albumentations
import albumentations.pytorch as AT
import torch
import torch.nn as nn
import torchvision.models as models
from plots import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = pd.read_csv('train.csv')
class_names = list(train['labels'].value_counts().sort_index().index)

train, valid = model_selection.train_test_split(train, stratify=train.labels, test_size=0.1, random_state=2020)

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('model.pth'))
model = model.to(device)
model.eval()

transform = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.CenterCrop(224, 224),
    albumentations.Normalize(),
    AT.ToTensor()
])

train_predictions = []
valid_predictions = []

for name, label in tqdm.tqdm(zip(train.image_id.values, train.labels.values), total=len(train)):
    image_path = 'G:\\Plant Pathology 2020\\images' + '\\' + name + '.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)['image']
    image = torch.unsqueeze(image, 0)
    image = image.to(device)

    output = model(image)
    _, pred = torch.max(output, 1)
    pred = class_names[pred[0]]
    train_predictions.append(pred)
for name, label in tqdm.tqdm(zip(valid.image_id.values, valid.labels.values), total=len(valid)):
    image_path = 'G:\\Plant Pathology 2020\\images' + '\\' + name + '.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)['image']
    image = torch.unsqueeze(image, 0)
    image = image.to(device)

    output = model(image)
    _, pred = torch.max(output, 1)
    pred = class_names[pred[0]]
    valid_predictions.append(pred)

print("Result\n" + "======\n")
print("Train Acc: {:.2f} %".format(metrics.accuracy_score(train.labels.values, train_predictions) * 100))
print("Valid Acc: {:.2f} %".format(metrics.accuracy_score(valid.labels.values, valid_predictions) * 100))

train_confusion_matrix = metrics.confusion_matrix(train.labels.values, train_predictions)
valid_confusion_matrix = metrics.confusion_matrix(valid.labels.values, valid_predictions)

plot_confusion_matrix(train_confusion_matrix, class_names)
plot_confusion_matrix(valid_confusion_matrix, class_names)

with open('history.json', 'r') as f:
    history = json.load(f)

plot_lr_curve(history)

plot_loss_curve(history)
plot_acc_curve(history)