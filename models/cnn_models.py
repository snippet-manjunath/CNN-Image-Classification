"""CNN model architectures for image classification."""

import torch
import torch.nn as nn
from config import MODEL_CONFIG

class SimpleNet(nn.Module):
    """Simple CNN with 1 convolutional layer."""
    
    def __init__(self, input_channels=1, num_classes=10, dropout_rate=0.5, use_batch_norm=True):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Fully connected layers - size depends on input
        # For 28x28: after conv(28x28) -> pool(14x14) -> 32*14*14 = 6272
        # For 32x32: after conv(32x32) -> pool(16x16) -> 32*16*16 = 8192
        self.fc_size = 32 * 14 * 14  # Default for MNIST, adjust for CIFAR-10
        
        self.fc1 = nn.Linear(self.fc_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)) if self.use_batch_norm else self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MediumNet(nn.Module):
    """Medium CNN with 2 convolutional layers."""
    
    def __init__(self, input_channels=1, num_classes=10, dropout_rate=0.5, use_batch_norm=True):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # Conv block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        self.fc_size = 64 * 7 * 7  # For MNIST: 28->14->7
        
        self.fc1 = nn.Linear(self.fc_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)) if self.use_batch_norm else self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.relu(self.bn2(self.conv2(x)) if self.use_batch_norm else self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepNet(nn.Module):
    """Deep CNN with 4 convolutional layers."""
    
    def __init__(self, input_channels=1, num_classes=10, dropout_rate=0.5, use_batch_norm=True):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # Conv block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Conv block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        if use_batch_norm:
            self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Conv block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        if use_batch_norm:
            self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        self.fc_size = 256 * 1 * 1  # For MNIST: 28->14->7->3->1
        
        self.fc1 = nn.Linear(self.fc_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)) if self.use_batch_norm else self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.relu(self.bn2(self.conv2(x)) if self.use_batch_norm else self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.relu(self.bn3(self.conv3(x)) if self.use_batch_norm else self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = self.relu(self.bn4(self.conv4(x)) if self.use_batch_norm else self.conv4(x))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class WideNet(nn.Module):
    """Wide CNN with larger filters."""
    
    def __init__(self, input_channels=1, num_classes=10, dropout_rate=0.5, use_batch_norm=True):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # Conv block 1 - 5x5 kernel
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, padding=2)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Conv block 2 - 5x5 kernel
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Conv block 3 - 3x3 kernel
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        if use_batch_norm:
            self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        self.fc_size = 256 * 3 * 3  # For MNIST: 28->14->7->3
        
        self.fc1 = nn.Linear(self.fc_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)) if self.use_batch_norm else self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.relu(self.bn2(self.conv2(x)) if self.use_batch_norm else self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.relu(self.bn3(self.conv3(x)) if self.use_batch_norm else self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model(architecture, input_channels=1, num_classes=10, **kwargs):
    """Get model by name."""
    models = {
        'SimpleNet': SimpleNet,
        'MediumNet': MediumNet,
        'DeepNet': DeepNet,
        'WideNet': WideNet,
    }
    
    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(models.keys())}")
    
    return models[architecture](input_channels, num_classes, **kwargs)
