# Assume `train_dl`, `valid_dl`, and `test_dl` are already created from preprocessing.
# Step 1: Define the Residual CNN Model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.identity_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        out += identity
        return F.relu(out)
class FCResCNN(nn.Module):
    def __init__(self, num_channels, sequence_length, num_classes):
        super(FCResCNN, self).__init__()
        self.down_sample = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=1, stride=1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
        )
        self.res_block1 = ResidualBlock(32, 32)
        self.res_block2 = ResidualBlock(32, 64)
        self.res_block3 = ResidualBlock(64, 128)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * sequence_length, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.down_sample(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.classifier(x)
        return x
    
    # Step 2: Model Configuration
print("Extracting model configuration...")
example_batch = next(iter(train_dl))
features, labels = example_batch
num_channels = features.shape[1]
sequence_length = features.shape[2]
num_classes = labels.shape[1]
print(f"Number of channels: {num_channels}")
print(f"Sequence length: {sequence_length}")
print(f"Number of classes: {num_classes}")
# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = FCResCNN(num_channels, sequence_length, num_classes).to(device)
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)