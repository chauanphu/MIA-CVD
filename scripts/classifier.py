import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from common import MedicalImageData, AssymetricLoss
from torchvision.transforms import v2
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader

# 1. IMPORT WANDB
import wandb

# 2. DEFINE HYPERPARAMETERS (Organize them in a dict)
ARCHITECTURE = "DenseNet121"
hyperparameters = {
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 64,
    "architecture": ARCHITECTURE,
    "loss_func": "BCEWithLogitsLoss",
    "resolution": "256x256",
    "dataset": "ChesX-ray8"
}

# 3. INITIALIZE WANDB
wandb.init(
    project="multi-label-experiment", # Group runs together
    name="EXP-003-C",          # Name of this specific run
    config=hyperparameters            # Pass the config for tracking
)

# Read the data
DATA_PATH = 'dataset/sample/sample_labels.csv'
IMAGE_PATH = 'dataset/sample/images/'

df = pd.read_csv(DATA_PATH)
# Count occurrences of each label
label_counts = {}
for row in df['Finding Labels']:
    for label in row.split('|'):
        label = label.strip()
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1

patient_ids = df["Patient ID"].nunique()
print(f"Number of unique patients: {patient_ids}")

train_ids, test_ids = train_test_split(
    df["Patient ID"].unique(), test_size=0.2, random_state=42
)

# Create datasets
transform = v2.Compose([
    v2.Resize(size=(256, 256), antialias=True),
    v2.ToDtype(torch.float32, scale=True), # Converts [0-255] to [0.0-1.0],
])

train_set = MedicalImageData(
    image_path=IMAGE_PATH,
    annot_paths=DATA_PATH,
    ids=train_ids,
    transform=transform
)
test_set = MedicalImageData(
    image_path=IMAGE_PATH,
    annot_paths=DATA_PATH,
    ids=test_ids,
    transform=transform
)

train_loader = DataLoader(train_set, batch_size=wandb.config.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=wandb.config.batch_size, shuffle=False)

# Mock Model & Data Setup

## SETTING UP THE MODEL
# Load pretrained ResNet50 and modify for multi-label classification
model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(label_counts))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# 4. WATCH THE MODEL (Optional)
# This logs gradients and parameters to help debug vanishing/exploding gradients
wandb.watch(model, log="all")

model.train()
# --- TRAINING LOOP ---
for epoch in range(wandb.config.epochs):
    batch_loss = 0.0
    batch_f1 = 0.0
    batch_accuracy = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- CALCULATE METRICS ---
        # Convert logits to binary predictions
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float()
        
        # Calculate F1 (Macro) using CPU/Numpy
        train_f1 = f1_score(labels.cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0)
        accuracy = accuracy_score(labels.cpu().numpy(), preds.detach().cpu().numpy())
        batch_loss += loss.item()
        batch_f1 += train_f1
        batch_accuracy += accuracy
    # AVERAGE LOSS OVER EPOCH
    loss = batch_loss / len(train_loader)
    f1 = batch_f1 / len(train_loader)
    accuracy = batch_accuracy / len(train_loader)
    # 5. LOG METRICS
    # Pass a dictionary of what you want to track
    wandb.log({
        "epoch": epoch,
        "train_loss": loss,
        "train_f1": f1,
        "train_accuracy": accuracy,
        "learning_rate": optimizer.param_groups[0]['lr']
    })
    # For every 10 epochs,
    print(f"Epoch {epoch}: Loss {loss:.4f}")
# Save the model at the end of training
torch.save(model.state_dict(), f"weights/{ARCHITECTURE.lower()}_model.pth")
# 6. FINISH THE RUN
# Essential in Jupyter Notebooks to signal the run is over
wandb.finish()