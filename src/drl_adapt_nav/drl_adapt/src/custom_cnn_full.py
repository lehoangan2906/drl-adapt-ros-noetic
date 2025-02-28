#!usr/bin/python

"""
Defines the Custom CNN architecture for the DRL-Adapt policy in the project.

Processes pedestrian kinematics (80x80x2), lidar scans, and sub-goal data into features
for navigation in crowded scenes, using a ResNet-based feature extractor.
"""

import os
import gym
import torch
import random
import numpy as np
import numpy.matlib
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

SEED1 = 1337    # Seed for random number generation

def set_seed(seed):
    """Set random seeds for deterministic results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# Helper functions
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


"""Bottleneck block for feature extraction."""
class Bottleneck(nn.Module):
    expansion = 2   # Expansion factor for output channels

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)  # 1x1 conv to reduce channels
        self.bn1 = norm_layer(width)           # Batch normalization
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  # 3x3 conv with stride
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)  # 1x1 conv to expand channels
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)      # ReLU activation
        self.downsample = downsample           # Downsample layer if needed
        self.stride = stride                   # Stride for convolution

    
    def forward(self, x):
        """Forward pass through bottleneck block."""
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity  # Residual connection
        out = self.relu(out)
        return out
    

class CustomCNN(BaseFeaturesExtractor):
    """Custom CNN architecture for extracting features from sensor and goal data."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        block = Bottleneck
        layers = [2, 1, 1]  # Number of bottleneck blocks per layer
        zero_init_residual = True
        groups = 1
        width_per_group = 64
        norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.inplanes = 64  # Initial input channels
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial conv layer for fused input (scan + ped_pos)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])  # First bottleneck layer
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # Second layer with downsampling
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # Third layer with downsampling
        
        # Additional conv blocks with residual connections
        self.conv2_2 = nn.Sequential(  # Refine features from layer2
            nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.downsample2 = nn.Sequential(  # Downsample for residual connection
            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2,2), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Sequential(  # Refine features from layer3
            nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.downsample3 = nn.Sequential(  # Downsample for residual connection
            nn.Conv2d(64, 512, kernel_size=(1, 1), stride=(4,4), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Pool to 1x1 feature map
        self.linear_fc = nn.Sequential(  # Final fully connected layer
            nn.Linear(256 * block.expansion + 2, features_dim),  # Combine fusion + goal features
            nn.ReLU()
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)


    """Create a sequence of bottleneck blocks."""
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, self.dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)
    

    """Process inputs through CNN layers."""
    def _forward_impl(self, ped_pos, scan, goal):
        # Fuse pedestrian and scan data
        ped_in = ped_pos.reshape(-1, 2, 80, 80)  # Reshape ped_pos_map to 80x80x2
        scan_in = scan.reshape(-1, 1, 80, 80)    # Reshape scan to 80x80x1
        fusion_in = torch.cat((scan_in, ped_in), dim=1)  # Concatenate to 80x80x3

        # Initial convolution and pooling
        x = self.conv1(fusion_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers with downsampling
        identity3 = self.downsample3(x)
        x = self.layer1(x)
        identity2 = self.downsample2(x)
        x = self.layer2(x)
        x = self.conv2_2(x)
        x += identity2
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.conv3_2(x)
        x += identity3
        x = self.relu3(x)

        # Pool and flatten fusion output
        x = self.avgpool(x)
        fusion_out = torch.flatten(x, 1)

        # Process goal input
        goal_in = goal.reshape(-1, 2)
        goal_out = torch.flatten(goal_in, 1)

        # Combine and output features
        fc_in = torch.cat((fusion_out, goal_out), dim=1)
        x = self.linear_fc(fc_in)
        return x


    """Extract observation components and forward through CNN."""
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        ped_pos = observations[:, :12800]  # First 12800 elements: ped_pos_map
        scan = observations[:, 12800:19200]  # Next 6400: processed scan
        goal = observations[:, 19200:]      # Last 2: goal_cart
        return self._forward_impl(ped_pos, scan, goal)