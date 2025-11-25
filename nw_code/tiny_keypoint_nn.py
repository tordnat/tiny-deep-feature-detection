"""
Tiny Keypoint Extractor (Student)
Input: 240x320 image
Output: 60x80 heatmap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyKeypointDetector(nn.Module):
  def __init__(self, output_size=(60, 80)):
    super().__init__()
    self.output_size = output_size

    # Encoder: 4x downsampling
    self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1) # 2x downsample
    self.bn1 = nn.BatchNorm2d(32)

    self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1) # 2x downsample
    self.bn2 = nn.BatchNorm2d(64)

    self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
    self.bn3 = nn.BatchNorm2d(64)

    # Detection head
    self.head_conv = nn.Conv2d(64, 32, 3, padding=1)
    self.head_out = nn.Conv2d(32, 1, 1)
  
  def forward(self, x):
    """
    Args:
        x: (B, 3, H, W) RGB image
    Returns:
        logits: (B, 1, H//4, W//4) keypoint detection logits
    """

    # Encode
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))

    # Detection head -> logits, not probabilities
    x = F.relu(self.head_conv(x))
    x = self.head_out(x)

    return x