import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class CountryClassifier(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = False):
        super().__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: [batch, 3, 224, 224]
        # Returns raw logits: [batch, num_classes]
        return self.backbone(x)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self(x)
        return logits.argmax(dim=1)
    
