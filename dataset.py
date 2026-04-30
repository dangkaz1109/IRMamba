import os
import glob
from PIL import Image, ImageOps, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import random

class GaussianBlur:
    """Apply Gaussian Blur with a given probability."""
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() <= self.prob:
            img = img.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(self.radius_min, self.radius_max)
                )
            )
        return img
    
class Solarization:
    """Apply Solarization with a given probability."""
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        

class DataAugmentationDINOForIR:
    """
    Heavy augmentation pipeline for DINO self-supervised pretraining,
    customized for Infrared images.
    """
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        # We use a custom mean/std for IR if you have it.
        # Using ImageNet defaults here as a placeholder.
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        # Infrared-safe color jitter (avoiding hue/saturation)
        color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0)

        # 1st global crop transform
        self.global_transfo1 = T.Compose([
            T.RandomResizedCrop(224, scale=global_crops_scale, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=9)], p=1.0),
            T.ToTensor(),
            self.normalize,
        ])

        # 2nd global crop transform (adding solarization)
        self.global_transfo2 = T.Compose([
            T.RandomResizedCrop(224, scale=global_crops_scale, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=9)], p=0.1),
            Solarization(p=0.2),
            T.ToTensor(),
            self.normalize,
        ])

        # Local crops transform
        self.local_crops_number = local_crops_number
        self.local_transfo = T.Compose([
            T.RandomResizedCrop(96, scale=local_crops_scale, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=9)], p=0.5),
            T.ToTensor(),
            self.normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    

def get_ad_transform(image_size=224):
    # Use a 1-channel mean and std
    normalize = T.Normalize(mean=[0.5], std=[0.5])

    return T.Compose([
        T.Resize(int(image_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        normalize,
    ])

def get_mask_transform(image_size=224):
    return T.Compose([
        T.Resize(int(image_size * 1.14), interpolation=T.InterpolationMode.NEAREST),
        T.CenterCrop(image_size),
        T.ToTensor()
    ])


class InfraredDataset(Dataset):
    def __init__(self, data_dir, phase='train', transform=None, mask_transform=None, extension="*.png"):
        """
        Args:
            data_dir (str): Root directory containing 'train', 'test', and 'GT'.
            phase (str): 'train' or 'test'.
            transform (callable): Transforms for the input image.
            mask_transform (callable): Transforms for the GT mask (usually just Resize + ToTensor).
        """
        self.data_dir = data_dir
        self.phase = phase
        self.transform = transform
        self.mask_transform = mask_transform

        self.image_paths = []
        self.labels = [] # 0 for good (normal), 1 for anomalous (defect)
        self.mask_paths = []

        phase_dir = os.path.join(data_dir, phase)

        if phase == 'train':
            good_dir = os.path.join(phase_dir, 'good')
            paths = glob.glob(os.path.join(good_dir, extension))
            self.image_paths.extend(paths)
            self.labels.extend([0] * len(paths))
            self.mask_paths.extend([None] * len(paths))

        elif phase == 'test':
            categories = os.listdir(phase_dir)
            for cat in categories:
                cat_dir = os.path.join(phase_dir, cat)
                if not os.path.isdir(cat_dir):
                    continue

                paths = glob.glob(os.path.join(cat_dir, extension))
                self.image_paths.extend(paths)

                if cat == 'good':
                    self.labels.extend([0] * len(paths))
                    self.mask_paths.extend([None] * len(paths))
                else:
                    self.labels.extend([1] * len(paths))
                    for p in paths:
                        filename = os.path.basename(p)
                        # Map to the GT directory.
                        # Note: In standard MVTec, masks sometimes have a "_mask" suffix.
                        # We check for the exact filename first, then fallback to _mask suffix.
                        gt_path = os.path.join(data_dir, 'GT', cat, filename)
                        if not os.path.exists(gt_path):
                            name, ext = os.path.splitext(filename)
                            gt_path = os.path.join(data_dir, 'GT', cat, f"{name}_mask{ext}")

                        self.mask_paths.append(gt_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # If testing, we must also return the label and mask
        if self.phase == 'test':
            if mask_path is not None and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
            else:
                # Generate a completely blank mask for 'good' test images
                orig_w, orig_h = Image.open(img_path).size
                mask = Image.new('L', (orig_w, orig_h), 0)

            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = T.ToTensor()(mask)

            return img, label, mask

        # If training (either DINO or AD), we usually just need the image
        return img
    

def build_dino_dataloader(data_dir, batch_size=32, num_workers=4):
    transform = DataAugmentationDINOForIR(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8
    )

    dataset = InfraredDataset(data_dir=data_dir, phase='train', transform=transform)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    return dataloader


def build_ad_dataloaders(data_dir, batch_size=2, num_workers=0, image_size=224):
    img_transform = get_ad_transform(image_size=image_size)
    mask_transform = get_mask_transform(image_size=image_size)

    train_dataset = InfraredDataset(
        data_dir=data_dir, phase='train', transform=img_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    test_dataset = InfraredDataset(
        data_dir=data_dir, phase='test', transform=img_transform, mask_transform=mask_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    return train_loader, test_loader