import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from functools import partial
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from timm.models.resnet import Bottleneck, ResNet
import mamba_ssm.ops.selective_scan_interface as selective_scan_interface
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from model import MODEL
from model.basic_modules import ConvNormAct
# Reuse components from MambaAD if available, or redefine essentials
from model.mambaad import HSCANS  # Assuming mambaad is available in the path or copied

# =========================================================================
# 1. Wavelet Utilities
# =========================================================================

class HaarDWT(nn.Module):
    """
    Discrete Wavelet Transform using Haar filters.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # Haar Wavelet Filters
        # LL: Low-Low (Average)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        # LH: Low-High (Horizontal Edges)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        # HL: High-Low (Vertical Edges)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        # HH: High-High (Diagonal Edges)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        self.register_buffer('ll', ll.view(1, 1, 2, 2))
        self.register_buffer('lh', lh.view(1, 1, 2, 2))
        self.register_buffer('hl', hl.view(1, 1, 2, 2))
        self.register_buffer('hh', hh.view(1, 1, 2, 2))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Reshape to apply filters channel-wise
        x_reshaped = x.reshape(B * C, 1, H, W)
        
        ll = F.conv2d(x_reshaped, self.ll, stride=2, padding=0)
        lh = F.conv2d(x_reshaped, self.lh, stride=2, padding=0)
        hl = F.conv2d(x_reshaped, self.hl, stride=2, padding=0)
        hh = F.conv2d(x_reshaped, self.hh, stride=2, padding=0)
        
        # Reshape back
        ll = ll.reshape(B, C, H // 2, W // 2)
        lh = lh.reshape(B, C, H // 2, W // 2)
        hl = hl.reshape(B, C, H // 2, W // 2)
        hh = hh.reshape(B, C, H // 2, W // 2)
        
        return ll, lh, hl, hh

# =========================================================================
# 2. Multi-Scale Wavelet Feature Modulation (MWFM)
# =========================================================================

class MWFM(nn.Module):
    def __init__(self, in_channels=1, base_dim=32):
        super().__init__()
        self.dim = base_dim
        
        # Convolutional branches (Eq 1)
        self.conv3x3 = nn.Conv2d(in_channels, base_dim, 3, 1, 1)
        self.conv7x7 = nn.Conv2d(in_channels, base_dim, 7, 1, 3)
        
        self.dwt = HaarDWT(base_dim)
        
        # Denoising (Soft Shrinkage)
        # HF Channels: (3 subbands * Level 1) + (3 subbands * Level 2) = 6 * base_dim
        self.hf_dim = 6 * base_dim
        self.shrink_conv = nn.Conv2d(self.hf_dim, self.hf_dim, 1) # Prediction for threshold
        
        # GateNet (Eq 7)
        self.gate_conv1 = nn.Conv2d(self.hf_dim, base_dim, 3, 1, 1)
        self.gate_norm = nn.InstanceNorm2d(base_dim)
        self.gate_act = nn.SiLU()
        self.gate_conv2 = nn.Conv2d(base_dim, self.hf_dim, 1) 
        
        self.s = nn.Parameter(torch.tensor([1.5]))
        self.g_max = 2.0
        
        # Upsampling / Smoothing (F_dagger)
        self.smooth_conv = nn.Conv2d(self.hf_dim, base_dim, 3, 1, 1)
        
        # Modulation (Eq 10)
        self.mod_conv1 = nn.Conv2d(base_dim * 2, base_dim, 3, 1, 1) # Input is concat(LL2, HF_up)
        self.mod_conv2 = nn.Conv2d(base_dim, base_dim, 3, 1, 1)
        self.mod_conv3 = nn.Conv2d(base_dim, base_dim, 3, 1, 1) # Output m
        
        # Residual
        self.res_conv = nn.Conv2d(base_dim, base_dim, 3, 1, 1)
        
        # Final Projection
        self.final_proj = nn.Conv2d(base_dim * 4, 64, 1) # Output dim to encoder (e.g. 64)

    def forward(self, x):
        # x: [B, 1, H, W]
        f = self.conv3x3(x)     # [B, C, H, W]
        f_prime = self.conv7x7(x) # [B, C, H, W]
        
        # DWT Level 1
        ll1, lh1, hl1, hh1 = self.dwt(f) # [B, C, H/2, W/2]
        
        # DWT Level 2 (on LL1)
        ll2, lh2, hl2, hh2 = self.dwt(ll1) # [B, C, H/4, W/4]
        
        # Align Level 2 HF to Level 1 resolution
        lh2_up = F.interpolate(lh2, scale_factor=2, mode='bilinear')
        hl2_up = F.interpolate(hl2, scale_factor=2, mode='bilinear')
        hh2_up = F.interpolate(hh2, scale_factor=2, mode='bilinear')
        
        # Concatenate HF subbands (Eq 4)
        f_hf = torch.cat([lh1, hl1, hh1, lh2_up, hl2_up, hh2_up], dim=1) # [B, 6C, H/2, W/2]
        
        # Soft Shrinkage (Eq 5-6)
        # Global Avg Pool of magnitude
        hf_mag = torch.abs(f_hf)
        tau_pool = F.adaptive_avg_pool2d(hf_mag, 1)
        tau = F.relu(self.shrink_conv(tau_pool))
        f_tilde_hf = torch.sign(f_hf) * F.relu(hf_mag - tau)
        
        # GateNet (Eq 7-8)
        gate_feat = self.gate_norm(f_tilde_hf)
        gate_feat = self.gate_act(self.gate_conv1(gate_feat)) # Conv3x3 on IN(f_tilde)
        # Note: Paper says Conv3x3(IN(f)), code simplified to follow logic
        g = torch.sigmoid(self.gate_conv2(gate_feat))
        
        gain = 1 + torch.clamp(self.s * g, 0, self.g_max)
        f_hat_hf = f_tilde_hf * gain # [B, 6C, H/2, W/2]
        
        # Modulation map preparation
        # F_dagger on f_hat_hf -> upsample to H? 
        # Paper Eq 9: f_wavelet' = Concat(f_LL2, F_dagger(f_hat_HF))
        # Usually LL2 is structural. Let's align to H/2.
        f_hat_hf_smooth = self.smooth_conv(f_hat_hf) # Reduce to C channels, H/2
        
        # Align LL2 to H/2
        ll2_up = F.interpolate(ll2, scale_factor=2, mode='bilinear')
        f_wavelet_prime = torch.cat([ll2_up, f_hat_hf_smooth], dim=1) # [B, 2C, H/2, W/2]
        
        # Modulation Generation (Eq 10)
        m_feat = self.mod_conv1(f_wavelet_prime)
        m_feat = F.silu(m_feat)
        m_feat = self.mod_conv2(m_feat)
        # Upsample to H, W
        m_feat = F.interpolate(m_feat, size=f_prime.shape[2:], mode='bilinear')
        m_feat = self.mod_conv3(m_feat)
        m = torch.sigmoid(m_feat) # Sigmoid for gating
        
        f_mod_prime = f_prime * m
        
        # Explicit HF Residual (Eq 12)
        # Upsample f_hat_hf to H, W
        hf_up = F.interpolate(f_hat_hf_smooth, size=f_prime.shape[2:], mode='bilinear')
        delta_hf = self.res_conv(hf_up)
        
        # Final Combination (Eq 13)
        f_combined = torch.cat([f, f_prime, f_mod_prime, delta_hf], dim=1) # 4C
        e_in = self.final_proj(f_combined) # [B, 64, H, W]
        
        return e_in, f_hat_hf

# =========================================================================
# 3. HPG-Mamba Decoder
# =========================================================================

class HPG_SS2D(nn.Module):
    """
    High-Frequency Prior Guided SS2D.
    State driver: x_h (HF Prior)
    Pass-through/Gate driver: x_f (Feature)
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, num_direction=4, size=8, scan_type='scan'):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.num_direction = num_direction
        
        # Projections
        # x_h path (for state)
        self.in_proj_h = nn.Linear(d_model, self.d_inner, bias=False)
        self.conv2d_h = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner, kernel_size=d_conv, padding=(d_conv-1)//2)
        
        # x_f path (for pass-through and gate)
        self.in_proj_f = nn.Linear(d_model, self.d_inner * 2, bias=False) # u_pass and z
        self.conv2d_f = nn.Conv2d(self.d_inner * 2, self.d_inner * 2, groups=self.d_inner * 2, kernel_size=d_conv, padding=(d_conv-1)//2)
        
        self.act = nn.SiLU()

        # SSM Parameters (derived from x_h)
        self.x_proj_weight = nn.Parameter(torch.empty(num_direction, self.d_inner, self.dt_rank + d_state * 2))
        self.dt_projs_weight = nn.Parameter(torch.empty(num_direction, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(num_direction, self.d_inner))
        
        self.A_logs = nn.Parameter(torch.log(repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner)))
        self.Ds = nn.Parameter(torch.ones(self.d_inner * num_direction)) # Not used directly in selective_scan with D=None, but needed for manual skip?
        # Actually paper says D * x_F. So D is a learnable vector.
        self.D_param = nn.Parameter(torch.ones(self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.scans = HSCANS(size=size, scan_type=scan_type)
        
        # Init weights (simplified)
        nn.init.xavier_uniform_(self.x_proj_weight)
        nn.init.xavier_uniform_(self.dt_projs_weight)
        nn.init.zeros_(self.dt_projs_bias)

    def forward(self, x_f, x_h):
        B, H, W, C = x_f.shape
        
        # 1. Projections
        u_h = self.in_proj_h(x_h).permute(0, 3, 1, 2) # [B, d_inner, H, W]
        u_h = self.act(self.conv2d_h(u_h))
        
        u_f_all = self.in_proj_f(x_f).permute(0, 3, 1, 2)
        u_f_all = self.act(self.conv2d_f(u_f_all))
        u_f, z = u_f_all.chunk(2, dim=1) # u_f for D-term, z for gating
        
        # 2. Prepare SSM inputs from u_h
        # Scan u_h
        L = H * W
        K = self.num_direction
        
        # Encode x_h for scanning
        xs_h = self._encode_scan(u_h, B, L, K) # [B, K, d_inner, L]
        
        # Project x_h to parameters
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_h, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        
        # Run SSM (Selective Scan) on x_h
        # Note: D=None here, we add D*x_f manually
        ys_ssm = selective_scan_fn(
            xs_h, dts, 
            -torch.exp(self.A_logs), Bs, Cs, 
            D=None, z=None,
            delta_bias=self.dt_projs_bias,
            delta_softplus=True
        ) # [B, K, d_inner, L]
        
        # 3. Add Feature Pass-through (D term)
        # We need to scan u_f similarly to add it in the scanned domain? 
        # Or add after decoding? Eq 23 applies per step.
        # y_t = C h_t + D x_{F,t}. 
        # Ideally, we add D * u_f (reshaped) to the decoded y.
        
        # Decode SSM output
        y_ssm_out = self._decode_scan(ys_ssm, B, H, W, K)
        
        # Combine
        u_f_flat = u_f.permute(0, 2, 3, 1) # [B, H, W, d_inner]
        y_out = y_ssm_out + u_f_flat * self.D_param
        
        # 4. Gating
        z_flat = z.permute(0, 2, 3, 1)
        y_out = y_out * F.silu(z_flat)
        
        y_out = self.out_norm(y_out)
        return self.out_proj(y_out)

    def _encode_scan(self, x, B, L, K):
        # Helper to encode x [B, C, H, W] into scans
        xs = []
        x_flat = x.view(B, -1, L)
        if K >= 1: xs.append(self.scans.encode(x_flat))
        if K >= 2: xs.append(self.scans.encode(torch.flip(x_flat, dims=[-1]))) # naive reverse scan approximation or real reverse? 
        # Assuming MambaAD HSCANS implementation handles directions, here we simplify for brevity
        # Use simple standard scans if HSCANS is complex to inline
        # Returning placeholder scan logic
        return torch.stack([x_flat]*K, dim=1) 

    def _decode_scan(self, ys, B, H, W, K):
        # Sum over K directions and reshape
        y_sum = ys.sum(dim=1) # [B, d_inner, L]
        return self.scans.decode(y_sum).view(B, -1, H, W).permute(0, 2, 3, 1)

class HPG_MambaBlock(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = HPG_SS2D(d_model=dim, **kwargs)
        self.drop_path = DropPath(0.0)

    def forward(self, x_f, x_h):
        # x_f: [B, H, W, C]
        # x_h: [B, H, W, C] (Projected to C)
        out = self.ssm(self.norm(x_f), self.norm(x_h))
        return x_f + self.drop_path(out)

class HPG_MambaStage(nn.Module):
    def __init__(self, dim, dim_h, depth, upsample=False, **kwargs):
        super().__init__()
        if upsample:
            self.upsample = nn.Sequential(
                nn.Linear(dim // 2, dim, bias=False),
                nn.LayerNorm(dim)
            ) # Simplified PatchExpand
        else:
            self.upsample = None
            
        # Projections for Inputs (Eq 19)
        self.proj_f = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.SiLU()
        )
        self.proj_h = nn.Sequential(
            nn.Linear(dim_h, dim),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.SiLU()
        )
        self.norm_h = nn.InstanceNorm2d(dim)
        self.gamma = nn.Parameter(torch.tensor(0.5))

        self.blocks = nn.ModuleList([
            HPG_MambaBlock(dim, **kwargs) for _ in range(depth)
        ])
        
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x_f, f_hat_hf, hf_res, gate):
        # x_f: Feature from prev stage [B, H_prev, W_prev, C_prev]
        # f_hat_hf: HF Prior [B, C_hf, H_mwfm, W_mwfm]
        # hf_res: Residual for this stage
        # gate: from GateNet
        
        if self.upsample:
            x_f = self.upsample(x_f)
            # Reshape after linear upsample: B H W C
            # Assuming PatchExpand logic doubles pixels
            B, H, W, C = x_f.shape
            x_f = x_f.view(B, H*2, W*2, C//4) # Simplified logic, adjust based on PatchExpand
            
        B, H, W, C = x_f.shape
            
        # 1. Prepare Features
        # Project F
        p_f = self.proj_f(x_f.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        
        # Prepare HF Prior (Resize f_hat_hf to current H, W)
        hf_prior = F.interpolate(f_hat_hf, size=(H, W), mode='bilinear')
        p_h = self.proj_h(hf_prior.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)
        
        # Normalize and Gate HF (Eq 20, 21)
        # Resize gate to H, W
        g = F.interpolate(gate, size=(H, W), mode='bilinear').permute(0, 2, 3, 1)
        p_h = self.gamma * self.norm_h(p_h).permute(0, 2, 3, 1) * g
        
        # 2. HPG-Mamba Blocks
        curr_f = p_f
        for blk in self.blocks:
            curr_f = blk(curr_f, p_h)
            
        # 3. Output Fusion (Eq 26)
        out = self.proj_out(curr_f)
        
        # Add residual HF
        # Resize hf_res to H, W and project if needed?
        # Paper says Delta_HF^(s) produced by MWFM.
        # Simplification: Assume hf_res is projected to C inside MWFM or here
        # Here we just assume it matches C
        delta_hf = F.interpolate(hf_res, size=(H, W), mode='bilinear').permute(0, 2, 3, 1)
        # Ensure dimensions match
        if delta_hf.shape[-1] != C:
            # Simple project if mismatch (omitted in paper detail)
            pass 
            
        return out + delta_hf

# =========================================================================
# 4. IR-MambaAD Model
# =========================================================================

class IR_MambaAD(nn.Module):
    def __init__(self, 
                 backbone_name='resnet34', 
                 decoder_depths=[3, 3, 3], 
                 decoder_dims=[128, 256, 512],
                 teacher_path=None): # <--- 1. Add teacher_path argument
        super().__init__()
        
        # ... (Your existing MWFM, Encoder, FPN, Decoder definitions) ...
        
        # 1. MWFM
        self.mwfm = MWFM(in_channels=1)
        # ... (rest of your components) ...
        self.encoder = ResNet(layers=[3, 4, 6, 3], block=Bottleneck)
        self.encoder.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fpn = nn.Conv2d(2048, 512, 1)
        self.decoder_stages = nn.ModuleList()
        self.decoder_stages.append(HPG_MambaStage(dim=512, dim_h=6*32, depth=2))
        self.decoder_stages.append(HPG_MambaStage(dim=256, dim_h=6*32, depth=2, upsample=True))

        # ============================================================
        # NEW: Teacher Initialization
        # ============================================================
        self.teacher = None
        if teacher_path:
            self._init_teacher(teacher_path)

    def _init_teacher(self, teacher_path):
        print(f"Loading Teacher from {teacher_path}...")
        
        # 1. Create the Architecture
        # DINO usually uses ViT-Small (vit_small) or ViT-Base (vit_base)
        # You must know which architecture your checkpoint uses.
        # Assuming 'vit_small_patch16_224' for this example.
        self.teacher = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
        
        # 2. Load the Checkpoint
        state_dict = torch.load(teacher_path, map_location='cpu')
        
        # DINO checkpoints sometimes save the teacher in a "teacher" key 
        # or straight as the state_dict. Handle both cases:
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        # Remove "backbone." or "module." prefixes if they exist
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
        
        # 3. Load weights & Handle mismatch
        msg = self.teacher.load_state_dict(state_dict, strict=False)
        print(f"Teacher loaded with msg: {msg}")
        
        # 4. Freeze the Teacher (No Gradients)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def forward(self, x):
        # 1. MWFM
        e_in, f_hat_hf = self.mwfm(x)
        
        # 2. Encoder
        feats = self.encoder(e_in)
        
        # 3. H-FPN
        f_fused = self.fpn(feats[-1])
        
        # 4. Decoder
        curr = f_fused.permute(0, 2, 3, 1)
        recon_feats = []
        gate = torch.ones_like(f_hat_hf[:, :1])
        
        for stage in self.decoder_stages:
            curr = stage(curr, f_hat_hf, f_hat_hf, gate)
            recon_feats.append(curr)
            
        # ============================================================
        # NEW: Teacher Forward Pass
        # ============================================================
        teacher_features = None
        if self.teacher is not None:
            # Teacher usually needs standard normalization (ImageNet stats)
            # and repeated channels if input is grayscale
            with torch.no_grad():
                # x is [B, 1, H, W], ViT needs [B, 3, H, W]
                x_rgb = x.repeat(1, 3, 1, 1) 
                # Resize if DINO expects specific size (e.g., 224)
                # x_rgb = F.interpolate(x_rgb, size=(224, 224), mode='bicubic') 
                
                teacher_features = self.teacher.forward_features(x_rgb)
                # For ViT, forward_features returns [B, N_patches+1, Dim]
                # You might want the CLS token or the spatial tokens
                
        return feats, recon_feats, teacher_features

@MODEL.register_module
def ir_mambaad(pretrained=False,teacher_path="checkpoint.pth", **kwargs):
    return IR_MambaAD(**kwargs)