import torch
import torch.nn as nn
import torch.optim as optim
from models.integrated_model import IntegratedAnomalyDetector
from dataset import build_ad_dataloaders

def load_and_freeze_model(model, pretrained_weights_path=None, device='cuda'):

    model = model.to(device)

    if pretrained_weights_path:
        print(f"Loading DINO pre-trained weights from {pretrained_weights_path}...")
        state_dict = torch.load(pretrained_weights_path, map_location=device)

        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully.")

    # 2. Freeze the MWFM and Encoder networks
    print("Freezing MWFM and Encoder components...")

    # Freeze Multi-Scale Wavelet Feature Modulation (MWFM)
    model.mwfm.eval() # Set to evaluation mode (affects BatchNorm/Dropout/InstanceNorm)
    for param in model.mwfm.parameters():
        param.requires_grad = False

    # Freeze ResNet-34 Encoder
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False

    # 3. Ensure H-FPN Bottleneck and HPG-Mamba Decoder remain trainable
    model.bottleneck.train()
    for param in model.bottleneck.parameters():
        param.requires_grad = True

    model.mamba_stage.train()
    for param in model.mamba_stage.parameters():
        param.requires_grad = True

    # Also ensure the upsampling path and final projection head are trainable
    model.upsample.train()
    for param in model.upsample.parameters():
        param.requires_grad = True

    model.head.train()
    for param in model.head.parameters():
        param.requires_grad = True

    # Print summary of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable Parameters (H-FPN & Decoder): {trainable_params:,}")
    print(f"Frozen Parameters (MWFM & Encoder): {frozen_params:,}")

    return model

def setup_decoder_optimizer(model, learning_rate=1e-4):

    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=1e-4)
    return optimizer

import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_anomaly_decoder(model, train_loader, optimizer, num_epochs=120, device='cuda'):
    model = model.to(device)
    criterion = nn.MSELoss()
    loss_history = []

    # Ensure memory allocator is optimized [cite: 1481]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache() # [cite: 1482]
    scaler = torch.amp.GradScaler('cuda') # [cite: 1484]

    print("Starting Reconstruction Training on Normal IR Samples...")

    for epoch in range(num_epochs):
        model.train()
        model.mwfm.eval()
        model.encoder.eval()

        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for images in loop:
            images = images.to(device) # [cite: 1494]

            if images.shape[1] == 3: # [cite: 1495]
                images = images.mean(dim=1, keepdim=True) # [cite: 1497, 1498]

            optimizer.zero_grad(set_to_none=True) # [cite: 1500]

            with torch.amp.autocast('cuda'):
                # The model's forward pass now correctly handles the parallel routing
                # and internal torch.no_grad() for the frozen parts!
                anomaly_map = model(images)
                target = torch.zeros_like(anomaly_map)
                loss = criterion(anomaly_map, target)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.6f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Avg MSE Loss: {avg_epoch_loss:.6f}")
        torch.cuda.empty_cache()

    print("Training Complete!")
    return loss_history

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anomaly_detector = IntegratedAnomalyDetector(d_model=64)
    data_load = build_ad_dataloaders("data/button_cell")
    train_loader, test_loader = data_load
    
    pretrained_path = 'data/dino_pretrain/dino_full_model_epoch30.pth'
    anomaly_detector = load_and_freeze_model(
        model=anomaly_detector,
        pretrained_weights_path=pretrained_path, # Replace with pretrained_path when ready
        device=device
    )
    optimizer = setup_decoder_optimizer(anomaly_detector)

    import matplotlib.pyplot as plt

    history = train_anomaly_decoder(
        model=anomaly_detector,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=120,
        device=device
    )
    
    save_path = "data/anomaly_decoder_final.pth" 
    torch.save(anomaly_detector.state_dict(), save_path)
    print(f"Đã lưu model tại {save_path}")

    plt.figure(figsize=(8, 4))
    plt.plot(history, marker='o', markersize=3)
    plt.title("HPG-Mamba Decoder Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()