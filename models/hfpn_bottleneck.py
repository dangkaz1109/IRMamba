import torch
import torch.nn as nn
import torch.nn.functional as F

class HalfFPNBottleneck(nn.Module):

    def __init__(self, in_channels_list=[64, 128, 256], out_channels=512):
        super().__init__()
        self.proj1 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.proj3 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)

        self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, feat1, feat2, feat3):

        target_h = feat3.shape[2] // 2
        target_w = feat3.shape[3] // 2
        target_size = (target_h, target_w)

        p1 = F.adaptive_avg_pool2d(self.proj1(feat1), target_size)
        p2 = F.adaptive_avg_pool2d(self.proj2(feat2), target_size)
        p3 = F.adaptive_avg_pool2d(self.proj3(feat3), target_size)

        fused = p1 + p2 + p3

        out = self.fusion_conv(fused)
        out = self.norm(out)
        out = self.activation(out)

        return out