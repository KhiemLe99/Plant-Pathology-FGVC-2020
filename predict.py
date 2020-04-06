import cv2
import tqdm
import pandas as pd
import albumentations
import albumentations.pytorch as AT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = pd.read_csv('train.csv')
class_names = list(train['labels'].value_counts().sort_index().index)
num_classes = len(class_names)

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('resnet50.pth'))
model = model.to(device)
model.eval()

transform = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.CenterCrop(224, 224),
    albumentations.Normalize(),
    AT.ToTensor()
])

test = pd.read_csv('test.csv')
result = []
for name in tqdm.tqdm(test.image_id.values):
    image_path = 'G:\\Plant Pathology 2020\\images' + '\\' + name + '.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)['image']
    image = torch.unsqueeze(image, 0)
    image = image.to(device)

    output = model(image)
    prob = F.softmax(output, dim=1)[0]
    prob = list(prob.detach().cpu().numpy())
    result.append([name] + prob)

submission = pd.DataFrame(result, columns=['image_id'] + class_names) 
submission.to_csv('submission.csv', index=False)