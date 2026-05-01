import gc
import torch
import os
from dataset import build_dino_dataloader
from models.dino import DINOStudentTeacher, DINOLoss


def train_dino(data_dir, epochs=40, batch_size=8, accum_steps=16):
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Memory-Optimized DINO Training on {device} ---")
    gc.collect()
    torch.cuda.empty_cache()

    # Setup Dataloader
    dino_loader = build_dino_dataloader(data_dir, batch_size=batch_size)

    # Now it is safe to move to device
    model = DINOStudentTeacher().to(device)

    optimizer = torch.optim.AdamW(
        list(model.student_backbone.parameters()) + list(model.student_head.parameters()),
        lr=0.0005, weight_decay=1e-4
    )

    dino_loss = DINOLoss(
        out_dim=4096,
        ncrops=10,
        warmup_teacher_temp=0.04,
        teacher_temp=0.07,
        warmup_teacher_temp_epochs=30,
        nepochs=epochs
    ).to(device)

    momentum_schedule = torch.linspace(0.996, 1.0, epochs)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for it, crops in enumerate(dino_loader):
            crops = [c.mean(dim=1, keepdim=True).to(device) for c in crops]
            global_crops = crops[:2]

            with torch.amp.autocast('cuda'):
                student_out = model.forward_student(crops)
                with torch.no_grad():
                    teacher_out = model.forward_teacher(global_crops)

                loss = dino_loss(student_out, teacher_out, epoch)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (it + 1) % accum_steps == 0 or (it + 1) == len(dino_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.student_backbone.parameters(), max_norm=2.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                model.update_teacher(momentum_schedule[epoch].item())

            epoch_loss += loss.item() * accum_steps
            del student_out, teacher_out, loss

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss/len(dino_loader):.4f}")

    print("DINO Pretraining Complete.")
    return model

dino_model = train_dino('data/button_cell', epochs=80, batch_size=8, accum_steps=16)

save_dir = 'data/dino_pretrain'
os.makedirs(save_dir, exist_ok=True) # Creates the folder if it doesn't exist

backbone_path = os.path.join(save_dir, 'dino_student_backbone_epoch30.pth')
torch.save(dino_model.student_backbone.state_dict(), backbone_path)
print(f"✅ DINO Backbone successfully saved to: {backbone_path}")

full_model_path = os.path.join(save_dir, 'dino_full_model_epoch30.pth')
torch.save(dino_model.state_dict(), full_model_path)
print(f"✅ Full DINO checkpoint saved to: {full_model_path}")