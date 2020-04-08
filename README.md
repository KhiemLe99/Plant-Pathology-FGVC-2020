# **Plant Pathology 2020 - FGVC7**

## **Competition Webside:** [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7)

## **Dataset Overview:**
|class|images|
|:--|--:|
|healthy|516|
|multiple_diseases|91|
|rust|622|
|scab|592|
||Total: 1821|

## **Solutions:**

**Data Preprocessing and Augmentations:**

- Resize(256, 256)
- CenterCrop(224, 224)
- HorizontalFlip(), VerticalFlip()
- ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15)
- RandomBrightness()
- HueSaturationValue()
- Normalize()

**Models**
    
- [ResNet50](https://arxiv.org/abs/1512.03385)
- [SE ResNet50](https://arxiv.org/abs/1709.01507)

**Training Settings**

- Validation Size = 0.1
- Batch Size = 8
- Epochs = 40   
- Loss function: [Class-Balanced](https://arxiv.org/abs/1901.05555) Binary Cross-Entropy Loss (beta = 0.99)
- Regularization: Weight Decay (lambda = 0.001)
- Optimizer: SGD (lr=0.01, momentum=0.9)
- Learning Rate Scheduler: ReduceLROnPlateau (factor=0.1, patience=5)

## **Results**

||||
|---|---|---|
||ResNet50|SE ResNet50|
|Train Accuracy|98.17 %|99.63 %|
|Valid Accuracy|96.72 %|97.81 %|
|Train Confusion Matrix|<img src='plots/resnet50/Train Confusion Matrix.png'>|<img src='plots/se_resnet50/Train Confusion Matrix.png'>|
|Valid Confusion Matrix|<img src='plots/resnet50/Valid Confusion Matrix.png'>|<img src='plots/se_resnet50/Valid Confusion Matrix.png'>|
|Public Score|0.942|0.947|
|Private Score|||