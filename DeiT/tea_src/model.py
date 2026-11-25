
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from config import (
    NUM_CLASSES,
    PRETRAINED
)


# ============================================================
# 1. Teacher ResNet-18
# ============================================================

class TeacherResNet18(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED
    ):

        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = resnet18(
            weights=weights
        )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            num_classes
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        return self.backbone(x)


# ============================================================
# 2. Build Model
# ============================================================

def build_model():

    model = TeacherResNet18()

    return model
