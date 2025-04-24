# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import config # Use updated config

class HistopathModel(nn.Module):
    """ Custom CNN for Histopathology Image Classification """
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(HistopathModel, self).__init__()

        # Architecture uses BatchNorm and Dropout
        # Dropout rates from original file: 0, 0, 0.15, 0.2, 0.4

        # Block 1
        self.conv1 = nn.Conv2d(config.INPUT_CHANNELS, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0) # No dropout

        # Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0) # No dropout

        # Block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.15)

        # Block 4
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.2)

        # Calculate the input size for the fully connected layer dynamically
        self.fc_input_size = self._get_conv_output_size(config.IMG_SIZE)
        if self.fc_input_size <= 0:
             raise ValueError(f"Calculated fc_input_size is {self.fc_input_size}. Check IMG_SIZE ({config.IMG_SIZE}) and model pooling layers.")

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout5 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_conv_output_size(self, img_size):
        # Helper to calculate flattened size after conv/pool layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, config.INPUT_CHANNELS, img_size, img_size)
            # Simulate forward pass through conv/pool layers
            x = self.pool1(self.conv2(self.conv1(dummy_input))) # Simplified for size calc
            x = self.pool2(self.conv4(self.conv3(x)))
            x = self.pool3(self.conv6(self.conv5(x)))
            x = self.pool4(self.conv8(self.conv7(x)))
            output_size = x.numel()
            return output_size

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Block 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.dropout4(x)

        # Flatten the output for the fully connected layers
        # Use adaptive flattening to handle potential size mismatches gracefully
        x = x.view(x.size(0), -1)
        # Verify flattened size matches expected fc_input_size (optional debug)
        # if x.size(1) != self.fc_input_size:
        #    print(f"Warning: Flattened size {x.size(1)} != calculated fc_input_size {self.fc_input_size}")

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x) # Output raw logits

        return x