import torch
import torch.nn as nn
import torch.nn.functional as F

# 28x28x1 -> 62 classes
class CNNPlus(nn.Module):
    def __init__(self, num_classes: int = 62, p_drop_mid: float = 0.25, p_drop_fc: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
            nn.Dropout(p_drop_mid),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(inplace=True), nn.Dropout(p_drop_fc),
            nn.Linear(256, num_classes),
        )
        self._init_kaming()

    def _init_kaming(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
