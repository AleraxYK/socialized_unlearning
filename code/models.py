import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    # Convolutional Neural Network (CNN) for Image Classification

    This **CNN** class implements a convolutional neural network (CNN) architecture with 
    three convolutional layers, batch normalization, pooling, dropout, and fully connected layers. 
    It is designed for image classification tasks, typically for datasets like CIFAR-10.

    Args:
        - num_classes (int): Number of output classes for classification (default is 10).
        - batch_size (int): The batch size used for training (default is 64).

    Attributes:
        - conv1, conv2, conv3: Convolutional layers that process the input image.
        - bn1, bn2, bn3: Batch normalization layers to normalize the activations.
        - pool1, pool2, pool3: Pooling layers to reduce spatial dimensions.
        - dropout1, dropout2, dropout3: Dropout layers to prevent overfitting.
        - fc1, fc2: Fully connected layers for classification.

    Methods:
        - forward(x): Defines the forward pass of the network.
    """

    def __init__(self, num_classes=10, batch_size=64):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.1)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout3 = nn.Dropout(p=0.1)

        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        # Forward Pass

        The forward pass applies the following operations in sequence:
        1. Convolution -> Batch Normalization -> ReLU -> Pooling -> Dropout (conv1)
        2. Convolution -> Batch Normalization -> ReLU -> Pooling -> Dropout (conv2)
        3. Convolution -> Batch Normalization -> ReLU -> Adaptive Pooling -> Dropout (conv3)
        4. Flatten the output and apply fully connected layers (fc1, fc2).

        Args:
            - x (tensor): The input tensor representing the images.

        Returns:
            - x (tensor): The output tensor after the forward pass.
        """
        
        # Apply conv1, batch norm, relu, pooling, and dropout
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Apply conv2, batch norm, relu, pooling, and dropout
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Apply conv3, batch norm, relu, adaptive pooling, and dropout
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten the output
        x = x.view(-1, 64)

        # Apply fc1 and relu
        x = F.relu(self.fc1(x))
        
        # Apply fc2 (output layer)
        x = self.fc2(x)
        
        return x
