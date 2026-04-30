import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from models.integrated_model import IntegratedAnomalyDetector


def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    
def generate_and_visualize_anomaly_maps(model, test_loader, device='cuda', num_samples=3):
    print("--- Generating Anomaly Maps ---")
    model.eval()
    model.to(device)

    samples_shown = 0
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for images, labels, masks in test_loader:
            if samples_shown >= num_samples:
                break

            images = images.to(device)
            if images.shape[1] == 3:
                images = images.mean(dim=1, keepdim=True)

            raw_anomaly_map = model(images)

            anomaly_score = torch.pow(raw_anomaly_map, 2)

            img_np = images[0].cpu().squeeze().numpy()
            mask_np = masks[0].cpu().squeeze().numpy()
            score_np = anomaly_score[0].cpu().squeeze().numpy()
            score_np = gaussian_filter(score_np, sigma=4)

            score_min, score_max = score_np.min(), score_np.max()
            score_np = (score_np - score_min) / (score_max - score_min + 1e-8)
            ax = axes[samples_shown]

            ax[0].imshow(img_np, cmap='gray')
            ax[0].set_title(f"Original IR Image\nLabel: {'Anomalous' if labels[0].item() == 1 else 'Normal'}")
            ax[0].axis('off')

            ax[1].imshow(mask_np, cmap='gray')
            ax[1].set_title("Ground Truth Mask")
            ax[1].axis('off')
            im = ax[2].imshow(score_np, cmap='jet')
            ax[2].set_title("Predicted Anomaly Map")
            ax[2].axis('off')
            fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

            samples_shown += 1

    plt.tight_layout()
    plt.show()

# Đoạn code ở file evaluate.py
import torch
from models.integrated_model import IntegratedAnomalyDetector
from dataset import build_ad_dataloaders
# ... (import các hàm vẽ hình)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntegratedAnomalyDetector(d_model=64)
    weight_path = "data/anomaly_decoder_final.pth"
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    _, test_loader = build_ad_dataloaders(data_dir="data/button_cell")

    generate_and_visualize_anomaly_maps(
        model=model,
        test_loader=test_loader,
        device=device,
        num_samples=3
    )