import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
from src.data.data_loader import LandmarkDataset
from src.models.models import LSTMClassifier
from src.utils.utils import set_seed, save_model
from tqdm import tqdm

# Training configuration
BATCH_SIZE = 64  # train 64 samples at a time
EPOCHS = 20
LEARNING_RATE = 4e-3
SEQ_LEN = 384
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
TRAIN_CSV = 'src/data/metadata/train.csv'
DATA_DIR = 'dataset/'

# Set seed for reproducibility
set_seed(42)

# Load dataset
train_dataset = LandmarkDataset(TRAIN_CSV, DATA_DIR, seq_len=SEQ_LEN)
print(f"Train dataset: {train_dataset}")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, loss, optimizer
model = LSTMClassifier(
    input_dim=train_dataset.num_features,
    hidden_dim=128,
    num_layers=2,
    num_classes=train_dataset.num_classes
).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# How often to print an explicit log line (set to 1 to log every batch)
log_every = 1

def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        batch_losses = []
        # Use tqdm to show a progress bar and update per-batch loss
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (inputs, labels) in pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            batch_losses.append(loss_val)
            total_loss += loss_val

            avg_loss_so_far = total_loss / (batch_idx + 1)

            # update tqdm postfix so you can monitor loss for each batch live
            pbar.set_postfix({
                'batch_loss': f"{loss_val:.4f}",
                'avg_loss': f"{avg_loss_so_far:.4f}"
            })

            # optional explicit print every `log_every` batches (keeps console cleaner if desired)
            if (batch_idx + 1) % log_every == 0:
                # This line logs batch-level info to stdout (in addition to tqdm postfix)
                print(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)} - batch_loss: {loss_val:.4f} - avg_loss: {avg_loss_so_far:.4f}")

        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Epoch Loss: {epoch_loss:.4f}")

    checkpoint_name = 'asl_checkpoint_2.pth'
    save_model(model, checkpoint_name)
    print(f'Training complete. Model saved as {checkpoint_name}')

if __name__ == '__main__':
    train()
