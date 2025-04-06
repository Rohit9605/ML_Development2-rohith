import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as DL
import torch.nn.functional as F
import time

# Step 3: Training Function
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=4):
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.argmax(dim=1).to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        # Validation phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for features, labels in valid_loader:
                features, labels = features.to(device), labels.argmax(dim=1).to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds. "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Validation Loss: {valid_loss / len(valid_loader):.4f}"
        )