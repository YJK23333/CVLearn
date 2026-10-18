from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

import config


class OxfordPetSegmentationDataset(Dataset):
    """Oxford-IIIT Pet dataset for binary pet-vs-background segmentation."""

    def __init__(self, split="train", augment=False):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.split = split
        self.augment = augment and split == "train"

        # Download/read the official trainval and test splits.
        self.trainval_dataset = None
        if split in {"train", "val"}:
            self.base_dataset = OxfordIIITPet(
                root=str(config.DATASET_ROOT),
                split=config.TRAINVAL_SPLIT,
                target_types="segmentation",
                download=config.DOWNLOAD_DATASET,
            )
            total = len(self.base_dataset)
            val_size = max(1, int(round(total * config.VALIDATION_RATIO)))
            train_size = total - val_size

            generator = torch.Generator().manual_seed(config.RANDOM_SEED)
            train_subset, val_subset = random_split(
                self.base_dataset,
                [train_size, val_size],
                generator=generator,
            )
            self.indices = train_subset.indices if split == "train" else val_subset.indices
        else:
            self.base_dataset = OxfordIIITPet(
                root=str(config.DATASET_ROOT),
                split=config.OFFICIAL_TEST_SPLIT,
                target_types="segmentation",
                download=config.DOWNLOAD_DATASET,
            )
            self.indices = list(range(len(self.base_dataset)))

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _mask_to_binary(mask_image):
        # The official trimap uses values 1 (pet), 2 (background), 3 (boundary).
        # Here both pet and uncertain boundary are mapped to foreground.
        mask = np.asarray(mask_image, dtype=np.uint8)
        pet_values = np.array(config.PET_MASK_VALUES, dtype=np.uint8)
        binary = np.isin(mask, pet_values).astype(np.float32)
        return torch.from_numpy(binary).unsqueeze(0)

    def __getitem__(self, index):
        original_index = self.indices[index]
        image, mask = self.base_dataset[original_index]

        # Resize RGB image with bilinear interpolation.
        image = TF.resize(
            image,
            list(config.IMAGE_SIZE),
            interpolation=InterpolationMode.BILINEAR,
        )

        # Resize categorical mask with nearest-neighbor interpolation to preserve labels.
        mask = TF.resize(
            mask,
            list(config.IMAGE_SIZE),
            interpolation=InterpolationMode.NEAREST,
        )

        if self.augment and random.random() < config.HORIZONTAL_FLIP_PROBABILITY:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.to_tensor(image)
        mask = self._mask_to_binary(mask)

        # Image tensor: [3, H, W], float in [0, 1].
        # Mask tensor: [1, H, W], float in {0, 1}.
        return image, mask
