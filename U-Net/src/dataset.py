
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import config


class OxfordPetDataset(Dataset):

    def __init__(self, samples, image_size, train=False):
        self.samples = samples
        self.image_size = image_size
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # Resize image and mask separately
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST
        )

        # Data augmentation: horizontal flip
        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Image: [3, H, W], range [0, 1]
        image = TF.to_tensor(image)

        # Mask: 1 = pet, 2/3 = background
        mask = np.array(mask, dtype=np.uint8)

        mask = (mask == 1).astype(np.float32)

        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def read_split_file(file_path):

    samples = []

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            # Format:
            # image_id class_id species breed_id
            parts = line.split()

            image_id = parts[0]

            image_path = config.IMAGE_DIR / f"{image_id}.jpg"
            mask_path = config.TRIMAP_DIR / f"{image_id}.png"

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found: {mask_path}"
                )

            samples.append((image_path, mask_path))

    return samples


def get_dataloaders():

    # Read official trainval split
    trainval_samples = read_split_file(
        config.TRAINVAL_FILE
    )

    # Read official test split
    test_samples = read_split_file(
        config.TEST_FILE
    )

    # Reproducible train/validation split
    val_size = int(
        len(trainval_samples) * config.VAL_RATIO
    )

    train_size = len(trainval_samples) - val_size

    generator = torch.Generator().manual_seed(
        config.RANDOM_SEED
    )

    train_subset, val_subset = random_split(
        trainval_samples,
        [train_size, val_size],
        generator=generator
    )

    train_samples = [
        trainval_samples[i]
        for i in train_subset.indices
    ]

    val_samples = [
        trainval_samples[i]
        for i in val_subset.indices
    ]

    # Dataset
    train_dataset = OxfordPetDataset(
        train_samples,
        config.IMAGE_SIZE,
        train=True
    )

    val_dataset = OxfordPetDataset(
        val_samples,
        config.IMAGE_SIZE,
        train=False
    )

    test_dataset = OxfordPetDataset(
        test_samples,
        config.IMAGE_SIZE,
        train=False
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print("=" * 50)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print("=" * 50)

    return train_loader, val_loader, test_loader
