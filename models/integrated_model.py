import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_encoder import ResNet34Encoder
from mwfm import ModifiedMWFM
from hfpn_bottleneck import HalfFPNBottleneck
from mamba_stage import HPG_Mamba_Stage

class IntegratedAnomalyDetector(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()

        # 1. Main Semantic Encoder
        self.encoder = ResNet34Encoder(pretrained=False) # [cite: 290, 331]

        # 2. Half-FPN Bottleneck Fusion
        self.bottleneck = HalfFPNBottleneck(in_channels_list=[64, 128, 256], out_channels=512) # [cite: 614, 675]

        # 3. High-Frequency Prior Extractor
        self.mwfm = ModifiedMWFM(in_channels=1, embed_dim=d_model) # [cite: 457]

        # 4. Upsampling Decoder Path
        # Transforms the H/32 bottleneck (512ch) back to HxW (d_model ch)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False),
            nn.Conv2d(512, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.SiLU(inplace=True)
        )

        # 5. Hybrid SSM Decoder Stage
        self.mamba_stage = HPG_Mamba_Stage(d_model=d_model) # [cite: 928, 997]

        # 6. Final Anomaly Projection Head
        self.head = nn.Conv2d(d_model, 1, kernel_size=1)

    def forward(self, x):
        # 1. Memory-Safe Frozen Forward Passes
        with torch.no_grad():
            f1, f2, f3 = self.encoder(x)
            E_in, hf_s, g_s, delta_hf_s = self.mwfm(x)

        # 2. Trainable Bottleneck
        f_bot = self.bottleneck(f1, f2, f3)

        # 3. Spatial Alignment
        target_size = x.shape[-2:]
        hf_s_aligned = F.interpolate(hf_s, size=target_size, mode='bilinear')
        g_s_aligned = F.interpolate(g_s, size=target_size, mode='bilinear')
        delta_hf_aligned = F.interpolate(delta_hf_s, size=target_size, mode='bilinear')

        # 4. Decoder Upsampling
        f_s = self.upsample(f_bot)

        # 5. HPG Mamba Processing
        # (Note: If this still OOMs, you must replace the PyTorch PoC loop in HPG_SSM_Core
        # with the official `mamba_ssm` kernel you pip installed earlier!)
        f_hat = self.mamba_stage(f_s, hf_s_aligned, g_s_aligned, delta_hf_aligned)

        # 6. Final Prediction
        anomaly_map = self.head(f_hat)

        return anomaly_map