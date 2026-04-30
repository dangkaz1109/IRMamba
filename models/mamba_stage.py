import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class CrossScan2D(nn.Module):

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W

        scan1 = x.view(B, C, L).transpose(1, 2) # (B, L, C)
        scan2 = torch.flip(scan1, dims=[1])
        scan3 = x.transpose(2, 3).contiguous().view(B, C, L).transpose(1, 2)
        scan4 = torch.flip(scan3, dims=[1])

        return torch.stack([scan1, scan2, scan3, scan4], dim=1) # (B, 4, L, C)

class ReverseScan2D(nn.Module):

    def forward(self, y_seqs, H, W):
        B, _, L, C = y_seqs.shape

        y1 = y_seqs[:, 0].transpose(1, 2).view(B, C, H, W)
        y2 = torch.flip(y_seqs[:, 1], dims=[1]).transpose(1, 2).view(B, C, H, W)
        y3 = y_seqs[:, 2].transpose(1, 2).view(B, C, W, H).transpose(2, 3)
        y4 = torch.flip(y_seqs[:, 3], dims=[1]).transpose(1, 2).view(B, C, W, H).transpose(2, 3)
        y_aggregated = y1 + y2 + y3 + y4
        return y_aggregated

class HPG_SSM_Core(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.hf_proj = nn.Linear(d_model, d_model)
        self.mamba = Mamba(
            d_model=d_model,  # Model dimension
            d_state=d_state,  # SSM state expansion factor
            d_conv=4,         # Local convolution width
            expand=2,         # Block expansion factor
        )

    def forward(self, x_f, x_h):

        hf_gate = torch.sigmoid(self.hf_proj(x_h))
        modulated_input = x_f * hf_gate

        y = self.mamba(modulated_input)

        return y

class HPG_Mamba_Stage(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.proj_f = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.SiLU()
        )
        self.proj_h = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.SiLU()
        )

        self.inorm = nn.InstanceNorm2d(d_model)
        self.gamma = nn.Parameter(torch.tensor(0.5)) # initialized near 0.5 [cite: 553]

        self.cross_scan = CrossScan2D()
        self.reverse_scan = ReverseScan2D()

        self.ssm_cores = nn.ModuleList([HPG_SSM_Core(d_model) for _ in range(4)])

        self.out_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, F_s, HF_s, G_s, Delta_HF_s):
        B, C, H, W = F_s.shape

        P_f = self.proj_f(F_s)
        P_h = self.proj_h(HF_s)

        P_h_tilde = self.inorm(P_h) * G_s
        P_h_bar = self.gamma * P_h_tilde

        X_f = self.cross_scan(P_f)       # (B, 4, L, C)
        X_h = self.cross_scan(P_h_bar)   # (B, 4, L, C)

        Y_seqs = []
        for i in range(4):
            y_i = self.ssm_cores[i](X_f[:, i], X_h[:, i])
            y_i = self.ln(y_i)
            Y_seqs.append(y_i)

        Y_seqs_tensor = torch.stack(Y_seqs, dim=1) # (B, 4, L, C)

        F_tilde = self.reverse_scan(Y_seqs_tensor, H, W)

        F_hat = self.out_proj(F_tilde) + Delta_HF_s

        return F_hat