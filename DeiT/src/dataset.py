
import os
import random

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from config import (
    DATA_DIR,
    IMAGE_SIZE,
    VAL_SPLIT,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED
)


# ============================================================
# 1. Class Mapping
# ============================================================

def build_class_mapping(data_dir):

    annotations_dir = os.path.join(
        data_dir,
        "annotations"
    )

    all_names = set()

    for filename in ["trainval.txt", "test.txt"]:

        file_path = os.path.join(
            annotations_dir,
            filename
        )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split()

                if parts:
                    all_names.add(parts[0])

    class_names = sorted({
        name.rsplit("_", 1)[0]
        for name in all_names
    })

    class_to_idx = {
        name: idx
        for idx, name in enumerate(class_names)
    }

    if len(class_to_idx) != 37:

        raise ValueError(
            f"Expected 37 classes, "
            f"got {len(class_to_idx)}"
        )

    return class_to_idx


# ============================================================
# 2. Oxford Pet Dataset
# ============================================================

class OxfordPetDataset(Dataset):

    def __init__(
        self,
        data_dir,
        split="train",
        transform=None,
        val_split=0.10,
        seed=42
    ):

        self.data_dir = data_dir

        self.split = split

        self.transform = transform

        self.images_dir = os.path.join(
            data_dir,
            "images"
        )

        annotations_dir = os.path.join(
            data_dir,
            "annotations"
        )

        # ----------------------------------------------------
        # Shared class mapping
        # ----------------------------------------------------

        self.class_to_idx = build_class_mapping(
            data_dir
        )

        self.idx_to_class = {
            idx: name
            for name, idx in self.class_to_idx.items()
        }

        # ----------------------------------------------------
        # Load annotations
        # ----------------------------------------------------

        if split in ["train", "val"]:

            annotation_file = os.path.join(
                annotations_dir,
                "trainval.txt"
            )

        elif split == "test":

            annotation_file = os.path.join(
                annotations_dir,
                "test.txt"
            )

        else:

            raise ValueError(
                f"Unknown split: {split}"
            )

        with open(
            annotation_file,
            "r",
            encoding="utf-8"
        ) as f:

            samples = [
                line.strip().split()[0]
                for line in f
                if line.strip()
            ]

        # ----------------------------------------------------
        # Train / Validation Split
        # ----------------------------------------------------

        if split in ["train", "val"]:

            rng = random.Random(seed)

            indices = list(range(len(samples)))

            rng.shuffle(indices)

            val_size = int(
                len(samples) * val_split
            )

            val_indices = indices[:val_size]

            train_indices = indices[val_size:]

            selected_indices = (
                train_indices
                if split == "train"
                else val_indices
            )

            self.samples = [
                samples[i]
                for i in selected_indices
            ]

        else:

            self.samples = samples

    # ========================================================
    # Get Item
    # ========================================================

    def __getitem__(self, index):

        image_name = self.samples[index]

        image_path = os.path.join(
            self.images_dir,
            image_name + ".jpg"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        class_name = image_name.rsplit("_", 1)[0]

        label = self.class_to_idx[class_name]

        if self.transform is not None:

            image = self.transform(image)

        return image, label

    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(self.samples)


# ============================================================
# 3. Transforms
# ============================================================

def get_transforms():

    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.8, 1.0),
            ratio=(0.75, 1.3333)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(
            IMAGE_SIZE
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


# ============================================================
# 4. DataLoaders
# ============================================================

def build_dataloaders():

    train_transform, eval_transform = get_transforms()

    train_dataset = OxfordPetDataset(
        data_dir=DATA_DIR,
        split="train",
        transform=train_transform,
        val_split=VAL_SPLIT,
        seed=SEED
    )

    val_dataset = OxfordPetDataset(
        data_dir=DATA_DIR,
        split="val",
        transform=eval_transform,
        val_split=VAL_SPLIT,
        seed=SEED
    )

    test_dataset = OxfordPetDataset(
        data_dir=DATA_DIR,
        split="test",
        transform=eval_transform,
        val_split=VAL_SPLIT,
        seed=SEED
    )

    # --------------------------------------------------------
    # Verify Class Mapping
    # --------------------------------------------------------

    assert (
        train_dataset.class_to_idx
        == val_dataset.class_to_idx
    )

    assert (
        train_dataset.class_to_idx
        == test_dataset.class_to_idx
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )
