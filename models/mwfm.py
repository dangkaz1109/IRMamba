import torch
import torch.nn as nn
import torch.nn.functional as F

def haar_dwt_2d(x):

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) / 2.0
    lh = (-x00 - x01 + x10 + x11) / 2.0
    hl = (-x00 + x01 - x10 + x11) / 2.0
    hh = (x00 - x01 - x10 + x11) / 2.0

    return ll, lh, hl, hh

class F_dagger(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.smooth = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, target_size=None, scale_factor=None):
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        elif scale_factor is not None:
            x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        return self.smooth(x)

class MWFM(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, G_max=5.0):
        super().__init__()
        self.G_max = G_max
        self.conv3x3 = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.conv7x7 = nn.Conv2d(in_channels, embed_dim, kernel_size=7, padding=3)

        hf_channels = embed_dim * 6
        self.tau_conv = nn.Conv2d(hf_channels, hf_channels, kernel_size=1)

        self.inorm = nn.InstanceNorm2d(hf_channels)
        self.gate_conv1 = nn.Conv2d(hf_channels, hf_channels, kernel_size=3, padding=1)
        self.gate_conv2 = nn.Conv2d(hf_channels, hf_channels, kernel_size=1)

        self.s = nn.Parameter(torch.tensor(1.5))

        self.up_hf = F_dagger(hf_channels, embed_dim)

        self.modulation_conv1 = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1)
        self.modulation_conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.up_mod = F_dagger(embed_dim, embed_dim)

        self.delta_hf_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)


        self.proj_in = nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1)

    def forward(self, x_ir):
        f = self.conv3x3(x_ir)
        f_prime = self.conv7x7(x_ir)

        f_ll1, f_lh1, f_hl1, f_hh1 = haar_dwt_2d(f)
        f_ll2, f_lh2, f_hl2, f_hh2 = haar_dwt_2d(f_ll1)

        target_size = f_lh1.shape[-2:]
        f_lh2_up = F.interpolate(f_lh2, size=target_size, mode='bilinear')
        f_hl2_up = F.interpolate(f_hl2, size=target_size, mode='bilinear')
        f_hh2_up = F.interpolate(f_hh2, size=target_size, mode='bilinear')

        f_hf = torch.cat([f_lh1, f_hl1, f_hh1, f_lh2_up, f_hl2_up, f_hh2_up], dim=1)


        mag_pool = F.adaptive_avg_pool2d(torch.abs(f_hf), 1)
        tau = F.relu(self.tau_conv(mag_pool))
        f_hf_tilde = torch.sign(f_hf) * F.relu(torch.abs(f_hf) - tau)

        gate_feat = self.inorm(f_hf_tilde)
        gate_feat = F.silu(self.gate_conv1(gate_feat))
        g = torch.sigmoid(self.gate_conv2(gate_feat))

        clamp_val = torch.clamp(self.s * g, min=0, max=self.G_max)
        f_hf_hat = f_hf_tilde * (1 + clamp_val)

        f_hf_hat_up = self.up_hf(f_hf_hat, target_size=f_ll2.shape[-2:])
        f_wavelet_prime = torch.cat([f_ll2, f_hf_hat_up], dim=1)

        m = F.silu(self.modulation_conv1(f_wavelet_prime))
        m = self.modulation_conv2(m)
        m_up = self.up_mod(m, target_size=f_prime.shape[-2:])

        f_mod_prime = f_prime * m_up

        f_hf_res_up = self.up_hf(f_hf_hat, target_size=f_prime.shape[-2:])
        delta_hf = self.delta_hf_conv(f_hf_res_up)

        f_combined = torch.cat([f, f_prime, f_mod_prime, delta_hf], dim=1)
        E_in = self.proj_in(f_combined)

        return E_in
    
class ModifiedMWFM(MWFM):
    """
    Wraps the existing MWFM to expose the intermediate high-frequency priors
    required by the HPG-Mamba decoder stage.
    """
    def forward(self, x_ir):
        # 1. Dual Branches
        f = self.conv3x3(x_ir)
        f_prime = self.conv7x7(x_ir)

        # 2. Two-Level DWT
        f_ll1, f_lh1, f_hl1, f_hh1 = haar_dwt_2d(f)
        f_ll2, f_lh2, f_hl2, f_hh2 = haar_dwt_2d(f_ll1)

        target_size = f_lh1.shape[-2:]
        f_lh2_up = F.interpolate(f_lh2, size=target_size, mode='bilinear')
        f_hl2_up = F.interpolate(f_hl2, size=target_size, mode='bilinear')
        f_hh2_up = F.interpolate(f_hh2, size=target_size, mode='bilinear')
        f_hf = torch.cat([f_lh1, f_hl1, f_hh1, f_lh2_up, f_hl2_up, f_hh2_up], dim=1)

        # 3. Soft-Shrinkage Denoising
        mag_pool = F.adaptive_avg_pool2d(torch.abs(f_hf), 1)
        tau = F.relu(self.tau_conv(mag_pool))
        f_hf_tilde = torch.sign(f_hf) * F.relu(torch.abs(f_hf) - tau)

        # 4. GateNet Adaptive Gain
        gate_feat = self.inorm(f_hf_tilde)
        gate_feat = F.silu(self.gate_conv1(gate_feat))
        g = torch.sigmoid(self.gate_conv2(gate_feat))
        clamp_val = torch.clamp(self.s * g, min=0, max=self.G_max)
        f_hf_hat = f_hf_tilde * (1 + clamp_val)

        # 5. Upsampling and Integration
        f_hf_hat_up = self.up_hf(f_hf_hat, target_size=f_ll2.shape[-2:])
        f_wavelet_prime = torch.cat([f_ll2, f_hf_hat_up], dim=1)

        # 6. Nonlinear Excitation & Gating
        m = F.silu(self.modulation_conv1(f_wavelet_prime))
        m = self.modulation_conv2(m)
        m_up = self.up_mod(m, target_size=f_prime.shape[-2:])
        f_mod_prime = f_prime * m_up

        # 7. Explicit HF Residual
        f_hf_res_up = self.up_hf(f_hf_hat, target_size=f_prime.shape[-2:])
        delta_hf = self.delta_hf_conv(f_hf_res_up)

        # 8. Final Combination
        f_combined = torch.cat([f, f_prime, f_mod_prime, delta_hf], dim=1)
        E_in = self.proj_in(f_combined)

        # --- THE FIX: Channel Reduction ---
        # g currently has 384 channels. We reshape it to group the 6 subbands,
        # then take the mean to yield a 64-channel spatial confidence gate.
        B, _, H_g, W_g = g.shape
        g_reduced = g.view(B, 6, -1, H_g, W_g).mean(dim=1)

        return E_in, f_hf_res_up, g_reduced, delta_hf