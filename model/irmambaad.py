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
from model.mambaad import HSCANS  

class HaarDWT(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        self.register_buffer('ll', ll.view(1, 1, 2, 2))
        self.register_buffer('lh', lh.view(1, 1, 2, 2))
        self.register_buffer('hl', hl.view(1, 1, 2, 2))
        self.register_buffer('hh', hh.view(1, 1, 2, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.reshape(B * C, 1, H, W)
        
        ll = F.conv2d(x_reshaped, self.ll, stride=2, padding=0)
        lh = F.conv2d(x_reshaped, self.lh, stride=2, padding=0)
        hl = F.conv2d(x_reshaped, self.hl, stride=2, padding=0)
        hh = F.conv2d(x_reshaped, self.hh, stride=2, padding=0)
        ll = ll.reshape(B, C, H // 2, W // 2)
        lh = lh.reshape(B, C, H // 2, W // 2)
        hl = hl.reshape(B, C, H // 2, W // 2)
        hh = hh.reshape(B, C, H // 2, W // 2)
        
        return ll, lh, hl, hh

class MWFM(nn.Module):
    def __init__(self, in_channels=1, base_dim=32):
        super().__init__()
        self.dim = base_dim
        self.conv3x3 = nn.Conv2d(in_channels, base_dim, 3, 1, 1)
        self.conv7x7 = nn.Conv2d(in_channels, base_dim, 7, 1, 3)
        
        self.dwt = HaarDWT(base_dim)
        
        self.hf_dim = 6 * base_dim
        self.shrink_conv = nn.Conv2d(self.hf_dim, self.hf_dim, 1) 
        self.gate_conv1 = nn.Conv2d(self.hf_dim, base_dim, 3, 1, 1)
        self.gate_norm = nn.InstanceNorm2d(base_dim)
        self.gate_act = nn.SiLU()
        self.gate_conv2 = nn.Conv2d(base_dim, self.hf_dim, 1) 
        
        self.s = nn.Parameter(torch.tensor([1.5]))
        self.g_max = 2.0
        self.smooth_conv = nn.Conv2d(self.hf_dim, base_dim, 3, 1, 1)
    
        self.mod_conv1 = nn.Conv2d(base_dim * 2, base_dim, 3, 1, 1) 
        self.mod_conv2 = nn.Conv2d(base_dim, base_dim, 3, 1, 1)
        self.mod_conv3 = nn.Conv2d(base_dim, base_dim, 3, 1, 1) 
        
        self.res_conv = nn.Conv2d(base_dim, base_dim, 3, 1, 1)
        
        self.final_proj = nn.Conv2d(base_dim * 4, 64, 1) 

    def forward(self, x):
        f = self.conv3x3(x)     
        f_prime = self.conv7x7(x) 
        
        ll1, lh1, hl1, hh1 = self.dwt(f) 
        ll2, lh2, hl2, hh2 = self.dwt(ll1) 
        
        lh2_up = F.interpolate(lh2, scale_factor=2, mode='bilinear')
        hl2_up = F.interpolate(hl2, scale_factor=2, mode='bilinear')
        hh2_up = F.interpolate(hh2, scale_factor=2, mode='bilinear')
        
        f_hf = torch.cat([lh1, hl1, hh1, lh2_up, hl2_up, hh2_up], dim=1) 
        
        hf_mag = torch.abs(f_hf)
        tau_pool = F.adaptive_avg_pool2d(hf_mag, 1)
        tau = F.relu(self.shrink_conv(tau_pool))
        f_tilde_hf = torch.sign(f_hf) * F.relu(hf_mag - tau)
        
        gate_feat = self.gate_norm(f_tilde_hf)
        gate_feat = self.gate_act(self.gate_conv1(gate_feat)) 
        g = torch.sigmoid(self.gate_conv2(gate_feat))
        
        gain = 1 + torch.clamp(self.s * g, 0, self.g_max)
        f_hat_hf = f_tilde_hf * gain 
        
        f_hat_hf_smooth = self.smooth_conv(f_hat_hf) 
        
        ll2_up = F.interpolate(ll2, scale_factor=2, mode='bilinear')
        f_wavelet_prime = torch.cat([ll2_up, f_hat_hf_smooth], dim=1) 
        
        m_feat = self.mod_conv1(f_wavelet_prime)
        m_feat = F.silu(m_feat)
        m_feat = self.mod_conv2(m_feat)
        m_feat = F.interpolate(m_feat, size=f_prime.shape[2:], mode='bilinear')
        m_feat = self.mod_conv3(m_feat)
        m = torch.sigmoid(m_feat) 
        
        f_mod_prime = f_prime * m
        
        hf_up = F.interpolate(f_hat_hf_smooth, size=f_prime.shape[2:], mode='bilinear')
        delta_hf = self.res_conv(hf_up)
        
        f_combined = torch.cat([f, f_prime, f_mod_prime, delta_hf], dim=1) 
        e_in = self.final_proj(f_combined) 
        
        return e_in, f_hat_hf

class HPG_SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, num_direction=4, size=8, scan_type='scan'):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.num_direction = num_direction
        
        self.in_proj_h = nn.Linear(d_model, self.d_inner, bias=False)
        self.conv2d_h = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner, kernel_size=d_conv, padding=(d_conv-1)//2)
        
        self.in_proj_f = nn.Linear(d_model, self.d_inner * 2, bias=False) 
        self.conv2d_f = nn.Conv2d(self.d_inner * 2, self.d_inner * 2, groups=self.d_inner * 2, kernel_size=d_conv, padding=(d_conv-1)//2)
        
        self.act = nn.SiLU()

        self.x_proj_weight = nn.Parameter(torch.empty(num_direction, self.d_inner, self.dt_rank + d_state * 2))
        self.dt_projs_weight = nn.Parameter(torch.empty(num_direction, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(num_direction, self.d_inner))
        
        self.A_logs = nn.Parameter(torch.log(repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner)))
        self.Ds = nn.Parameter(torch.ones(self.d_inner * num_direction)) 
        self.D_param = nn.Parameter(torch.ones(self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.scans = HSCANS(size=size, scan_type=scan_type)
        
        nn.init.xavier_uniform_(self.x_proj_weight)
        nn.init.xavier_uniform_(self.dt_projs_weight)
        nn.init.zeros_(self.dt_projs_bias)

    def forward(self, x_f, x_h):
        B, H, W, C = x_f.shape
        
        u_h = self.in_proj_h(x_h).permute(0, 3, 1, 2) 
        u_h = self.act(self.conv2d_h(u_h))
        
        u_f_all = self.in_proj_f(x_f).permute(0, 3, 1, 2)
        u_f_all = self.act(self.conv2d_f(u_f_all))
        u_f, z = u_f_all.chunk(2, dim=1) 
        
        L = H * W
        K = self.num_direction
        
        xs_h = self._encode_scan(u_h, B, L, K) 
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_h, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        
        ys_ssm = selective_scan_fn(
            xs_h, dts, 
            -torch.exp(self.A_logs), Bs, Cs, 
            D=None, z=None,
            delta_bias=self.dt_projs_bias,
            delta_softplus=True
        ) 
        
        y_ssm_out = self._decode_scan(ys_ssm, B, H, W, K)
        
        u_f_flat = u_f.permute(0, 2, 3, 1) 
        y_out = y_ssm_out + u_f_flat * self.D_param
        
        z_flat = z.permute(0, 2, 3, 1)
        y_out = y_out * F.silu(z_flat)
        
        y_out = self.out_norm(y_out)
        return self.out_proj(y_out)

    def _encode_scan(self, x, B, L, K):
        xs = []
        x_flat = x.view(B, -1, L)
        if K >= 1: xs.append(self.scans.encode(x_flat))
        if K >= 2: xs.append(self.scans.encode(torch.flip(x_flat, dims=[-1]))) 
        return torch.stack([x_flat]*K, dim=1) 

    def _decode_scan(self, ys, B, H, W, K):
        y_sum = ys.sum(dim=1) 
        return self.scans.decode(y_sum).view(B, -1, H, W).permute(0, 2, 3, 1)

class HPG_MambaBlock(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = HPG_SS2D(d_model=dim, **kwargs)
        self.drop_path = DropPath(0.0)

    def forward(self, x_f, x_h):
        out = self.ssm(self.norm(x_f), self.norm(x_h))
        return x_f + self.drop_path(out)

class HPG_MambaStage(nn.Module):
    def __init__(self, dim, dim_h, depth, upsample=False, **kwargs):
        super().__init__()
        if upsample:
            self.upsample = nn.Sequential(
                nn.Linear(dim // 2, dim, bias=False),
                nn.LayerNorm(dim)
            ) 
        else:
            self.upsample = None
            
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
        if self.upsample:
            x_f = self.upsample(x_f)
            B, H, W, C = x_f.shape
            x_f = x_f.view(B, H*2, W*2, C//4) 
            
        B, H, W, C = x_f.shape
            
        p_f = self.proj_f(x_f.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        
        hf_prior = F.interpolate(f_hat_hf, size=(H, W), mode='bilinear')
        p_h = self.proj_h(hf_prior.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)
        
        g = F.interpolate(gate, size=(H, W), mode='bilinear').permute(0, 2, 3, 1)
        p_h = self.gamma * self.norm_h(p_h).permute(0, 2, 3, 1) * g
        
        curr_f = p_f
        for blk in self.blocks:
            curr_f = blk(curr_f, p_h)
            
        out = self.proj_out(curr_f)
        
        delta_hf = F.interpolate(hf_res, size=(H, W), mode='bilinear').permute(0, 2, 3, 1)
        if delta_hf.shape[-1] != C:
            pass 
            
        return out + delta_hf

class IR_MambaAD(nn.Module):
    def __init__(self, 
                 backbone_name='resnet34', 
                 decoder_depths=[3, 3, 3], 
                 decoder_dims=[128, 256, 512],
                 teacher_path=None): 
        super().__init__()
        
        self.mwfm = MWFM(in_channels=1)
        self.encoder = ResNet(layers=[3, 4, 6, 3], block=Bottleneck)
        self.encoder.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fpn = nn.Conv2d(2048, 512, 1)
        self.decoder_stages = nn.ModuleList()
        self.decoder_stages.append(HPG_MambaStage(dim=512, dim_h=6*32, depth=2))
        self.decoder_stages.append(HPG_MambaStage(dim=256, dim_h=6*32, depth=2, upsample=True))

        self.teacher = None
        if teacher_path:
            self._init_teacher(teacher_path)

    def _init_teacher(self, teacher_path):
        self.teacher = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
        
        state_dict = torch.load(teacher_path, map_location='cpu')
        
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
        
        self.teacher.load_state_dict(state_dict, strict=False)
        
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def forward(self, x):
        e_in, f_hat_hf = self.mwfm(x)
        
        feats = self.encoder(e_in)
        
        f_fused = self.fpn(feats[-1])
        
        curr = f_fused.permute(0, 2, 3, 1)
        recon_feats = []
        gate = torch.ones_like(f_hat_hf[:, :1])
        
        for stage in self.decoder_stages:
            curr = stage(curr, f_hat_hf, f_hat_hf, gate)
            recon_feats.append(curr)
            
        teacher_features = None
        if self.teacher is not None:
            with torch.no_grad():
                x_rgb = x.repeat(1, 3, 1, 1) 
                
                teacher_features = self.teacher.forward_features(x_rgb)
                
        return feats, recon_feats, teacher_features

@MODEL.register_module
def ir_mambaad(pretrained=False, teacher_path="checkpoint.pth", **kwargs):
    return IR_MambaAD(teacher_path=teacher_path, **kwargs)
