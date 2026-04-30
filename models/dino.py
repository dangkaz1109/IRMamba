import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_encoder import ResNet34Encoder
from mwfm import ModifiedMWFM



class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=4096, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.apply(self._init_weights)

        # FIX: Updated to the new PyTorch 2.x parametrizations API
        self.last_layer = torch.nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))

        # In the new API, original0 represents the magnitude (weight_g)
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        self.last_layer.parametrizations.weight.original0.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2) # L2 Normalization
        x = self.last_layer(x)
        return x
    


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Schedule for teacher temperature (sharpening)
        self.teacher_temp_schedule = torch.cat((
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        student_output: (B * ncrops, out_dim)
        teacher_output: (B * 2, out_dim) - Teacher only processes the 2 global crops
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops) # List of tensors, each (B, out_dim)

        # Centering and sharpening for the teacher
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2) # 2 global crops

        total_loss = 0
        n_loss_terms = 0

        # Calculate Cross-Entropy between Teacher's global crops and Student's multi-crops
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue # Skip comparing a global crop to itself

                # Cross-entropy: - \sum P_teacher * log(P_student)
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        # EMA update of the center: c <- m * c + (1 - m) * batch_mean
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)



class DINOStudentTeacher(nn.Module):
    def __init__(self, out_dim=4096):
        super().__init__()

        # Build Student
        self.student_backbone = CombinedEncoder()
        self.student_head = DINOHead(in_dim=self.student_backbone.out_dim, out_dim=out_dim)

        # Build Teacher
        self.teacher_backbone = CombinedEncoder()
        self.teacher_head = DINOHead(in_dim=self.teacher_backbone.out_dim, out_dim=out_dim)

        # Teacher does not require gradients
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # Match initial weights perfectly
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

    def forward_student(self, crops):
        # Process all crops (global + local) [cite: 151-152]
        embs = torch.cat([self.student_backbone(x) for x in crops], dim=0)
        return self.student_head(embs)

    @torch.no_grad()
    def forward_teacher(self, global_crops):
        # Process ONLY the 2 global crops [cite: 151]
        embs = torch.cat([self.teacher_backbone(x) for x in global_crops], dim=0)
        return self.teacher_head(embs)

    @torch.no_grad()
    def update_teacher(self, momentum):
        # Exponential Moving Average update for backbone
        for param_q, param_k in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

        # Exponential Moving Average update for head
        for param_q, param_k in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)