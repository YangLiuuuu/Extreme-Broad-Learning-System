import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self,input_dim):
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, 8, 5, 1, 2),
            # nn.InstanceNorm2d(8),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            # nn.Dropout(p=0.3),

            nn.Conv2d(8, 16, 3, 1, 1),
            # nn.InstanceNorm2d(16),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            
            # nn.Dropout(p=0.3),

            nn.Conv2d(16, 32, 3, 1, 1),
            # nn.InstanceNorm2d(32),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            # nn.Dropout(p=0.3),

            nn.Conv2d(32, 32, 3, 1, 1),
            # nn.InstanceNorm2d(64),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            # nn.Dropout(p=0.3),

            nn.Flatten(),
            nn.Linear(3200, 256),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.model(x)