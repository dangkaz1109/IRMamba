import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from models.integrated_model import IntegratedAnomalyDetector

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def evaluate_anomaly_metrics(model, test_loader, device='cuda'):
    print("Computing AUROC Metrics")
    model.eval()
    model.to(device)

    all_image_labels = []
    all_image_scores = []

    all_pixel_masks = []
    all_pixel_scores = []

    loop = tqdm(test_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for images, labels, masks in loop:
            images = images.to(device)

            if images.shape[1] == 3:
                images = images.mean(dim=1, keepdim=True)

            with torch.amp.autocast('cuda'):
                raw_anomaly_map = model(images)

            anomaly_scores = torch.pow(raw_anomaly_map, 2).float().cpu().numpy()
            masks = masks.cpu().numpy()
            labels = labels.numpy()

            for i in range(images.size(0)):
                score_map = anomaly_scores[i].squeeze()
                gt_mask = masks[i].squeeze()

                score_map = gaussian_filter(score_map, sigma=4)

                img_score = np.max(score_map)

                all_image_labels.append(labels[i])
                all_image_scores.append(img_score)

                gt_mask = (gt_mask > 0).astype(int)

                all_pixel_masks.extend(gt_mask.flatten())
                all_pixel_scores.extend(score_map.flatten())

    print("\nCalculating AUROC scores...")

    try:
        image_auroc = roc_auc_score(all_image_labels, all_image_scores)
        pixel_auroc = roc_auc_score(all_pixel_masks, all_pixel_scores)

        print("========================================")
        print(f"Image-level AUROC: {image_auroc:.4f}")
        print(f"Pixel-level AUROC: {pixel_auroc:.4f}")
        print("========================================")

        return image_auroc, pixel_auroc

    except ValueError as e:
        print(f"\nError calculating AUROC: {e}")
        print("This usually happens if your test set only contains 'good' images and no defects.")
        return None, None


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
    plt.savefig('eva_result.png', bbox_inches='tight')
    print("Saved as eva_result.png!")
    plt.close()

import torch
from models.integrated_model import IntegratedAnomalyDetector
from dataset import build_ad_dataloaders

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntegratedAnomalyDetector(d_model=64)
    weight_path = "data/anomaly_decoder_final.pth"
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    _, test_loader = build_ad_dataloaders(data_dir="data/button_cell")
    img_auc, pix_auc = evaluate_anomaly_metrics(
        model=model,
        test_loader=test_loader,
        device=device
    )
    generate_and_visualize_anomaly_maps(
        model=model,
        test_loader=test_loader,
        device=device,
        num_samples=10
    )
