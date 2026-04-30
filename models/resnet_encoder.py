import torch
import torch.nn as nn
import torchvision.models as models


class ResNet34Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet34Encoder, self).__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        return f1, f2, f3

if __name__ == "__main__":
    dummy_input = torch.randn(2, 1, 256, 256)

    encoder = ResNet34Encoder(pretrained=False)
    feat1, feat2, feat3 = encoder(dummy_input)

    print(f"Feature 1 shape: {feat1.shape}") # Expected: [2, 64, 64, 64]
    print(f"Feature 2 shape: {feat2.shape}") # Expected: [2, 128, 32, 32]
    print(f"Feature 3 shape: {feat3.shape}") # Expected: [2, 256, 16, 16]