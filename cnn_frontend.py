import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width))

    def forward(self, x):
        return F.relu(x + self.branch(x))

class PaperCNN(nn.Module):

    def __init__(self, width=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(3)))
        self.head = nn.Conv2d(width, 2, 3, padding=1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        raw = self.head(self.blocks(self.stem(x)))
        return (raw[:, 0], raw[:, 1])
