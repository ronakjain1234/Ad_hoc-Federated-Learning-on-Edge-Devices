# src/adhocfl/models/cnn_cifar.py
import torch.nn as nn

class CIFAR10Small(nn.Module):
    # ~0.7M params; fast on laptop GPUs
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 8x8
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

class CIFAR10LiteRes(nn.Module):
    """
    ~1.2M params. 3 stages, light residual connections, GAP head.
    32x32 -> (64)x32 -> (128)x16 -> (256)x8 -> GAP -> FC(10)
    """
    def __init__(self, num_classes: int = 10, drop=0.25):
        super().__init__()
        self.stem = _ConvBNReLU(3, 64, k=3, s=1, p=1)

        # Stage 1 (no downsample)
        self.b1a = _ConvBNReLU(64, 64)
        self.b1b = _ConvBNReLU(64, 64)
        # Stage 2 (downsample)
        self.down2 = _ConvBNReLU(64, 128, s=2)   # 16x16
        self.b2a = _ConvBNReLU(128, 128)
        self.b2b = _ConvBNReLU(128, 128)
        # Stage 3 (downsample)
        self.down3 = _ConvBNReLU(128, 256, s=2)  # 8x8
        self.b3a = _ConvBNReLU(256, 256)
        self.b3b = _ConvBNReLU(256, 256)

        self.dropout = nn.Dropout(drop)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 256x1x1
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def _res(self, x, a, b):
        out = a(x)
        out = b(out)
        return nn.functional.relu(out + x, inplace=True)

    def forward(self, x):
        x = self.stem(x)
        x = self._res(x, self.b1a, self.b1b)
        x = self.down2(x)
        x = self._res(x, self.b2a, self.b2b)
        x = self.down3(x)
        x = self._res(x, self.b3a, self.b3b)
        x = self.dropout(x)
        return self.head(x)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out

def _make_layer(in_planes, planes, num_blocks, stride):
    layers = [BasicBlock(in_planes, planes, stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(planes, planes, 1))
    return nn.Sequential(*layers)

class ResNet20CIFAR(nn.Module):
    """ Depth 20 = 3 stages × (3 blocks each) with downsample at stage starts. """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.layer1 = _make_layer(16, 16, 3, 1)   # 32x32
        self.layer2 = _make_layer(16, 32, 3, 2)   # 16x16
        self.layer3 = _make_layer(32, 64, 3, 2)   # 8x8
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

