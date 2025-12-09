from torch import nn
import torch
from torchvision.io import read_image
import os
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd

# Create a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AssymetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super(AssymetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        # x: input logits, y: targets (multi-hot)
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross Entropy
        los_pos = y * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=1e-8))

        # Asymmetric Focusing
        loss = - (self.gamma_pos * los_pos + self.gamma_neg * los_neg)
        return loss.sum()
    
class MedicalImageData(torch.utils.data.Dataset):
    def __init__(self, image_path: str, annot_paths: str, transform=None, ids=None):
        self.img_labels = pd.read_csv(annot_paths)
        self.image_path = image_path
        self.transform = transform

        # One-hot encode the multi-labels
        labels_list = []
        for row in self.img_labels['Finding Labels']:
            labels = [label.strip() for label in row.split('|') if label.strip()]
            labels_list.append(labels)
        mlb = MultiLabelBinarizer()
        one_hot_encoded = mlb.fit_transform(labels_list)
        self.label_mapping = mlb.classes_
        self.labels_encoded = np.asarray(one_hot_encoded)
        self.patient_ids = self.img_labels['Patient ID'].nunique()

        # Only keep data for specified ids if provided
        if ids is not None:
            mask = self.img_labels['Patient ID'].isin(ids)
            self.img_labels = self.img_labels[mask].reset_index(drop=True)
            self.labels_encoded = self.labels_encoded[mask.values]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Load the image
        image_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.image_path, str(image_name))
        image = read_image(img_path)
        
        # Convert to RGB if necessary (handle grayscale or RGBA)
        if image.ndim == 2:  # Grayscale 2D image
            image = image.unsqueeze(0)  # Add channel dimension
        
        if image.shape[0] == 1:  # Grayscale (1 channel)
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:  # RGBA (4 channels)
            image = image[:3]
        elif image.shape[0] > 3:  # More than 3 channels
            image = image[:3]
        
        if self.transform:
            image = self.transform(image)
        # Get the multi-labels
        labels = self.labels_encoded[idx]
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels