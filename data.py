import cv2
import sklearn.preprocessing as preprocessing
import torch.utils as utils

def prepare_labels(df):
    labels = df['labels']

    label_encoder = preprocessing.LabelEncoder()
    label_encoded = label_encoder.fit_transform(labels)
    label_encoded = label_encoded.reshape(len(label_encoded), 1)

    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(label_encoded)

    return onehot_encoded

def smooth_labels(labels, factor=0):
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])

    return labels

class LeafDataset(utils.data.Dataset):
    def __init__(self, df, image_dir, transform, smooth_factor=0):
        self.image_files_list = [image_dir + '//' + i + '.jpg' for i in df['image_id'].values]
        self.transform = transform
        self.labels = prepare_labels(df)
        if smooth_factor > 0:
            self.labels = smooth_labels(self.labels, smooth_factor)

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        image_path = self.image_files_list[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']

        label = self.labels[idx]

        return image, label