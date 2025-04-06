# Step 4: Evaluation Function
def evaluate_model(model, test_loader):
    print("Evaluating model on test data...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            features, labels = features.to(device), labels.argmax(dim=1).to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if batch_idx % 10 == 0:  # Print progress every 10 batches
                print(f"  Batch {batch_idx}/{len(test_loader)}, Running Accuracy: {100 * correct / total:.2f}%")
    accuracy = 100 * correct / total
    print(f"Final Test Accuracy: {accuracy:.2f}%")
# Step 5: Run Training and Evaluation
train_model(model, train_dl, valid_dl, criterion, optimizer, num_epochs=4)
evaluate_model(model, test_dl)