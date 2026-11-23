
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import config


# ============================================================
# Configuration
# ============================================================

FOREGROUND_VALUES = getattr(
    config,
    "FOREGROUND_VALUES",
    (1, 3),
)

HORIZONTAL_FLIP_PROB = getattr(
    config,
    "HORIZONTAL_FLIP_PROB",
    0.5,
)


# ============================================================
# Path helpers
# ============================================================

def get_image_path(image_id):

    return os.path.join(
        config.DATASET_ROOT,
        "images",
        f"{image_id}.jpg",
    )


def get_trimap_path(image_id):

    return os.path.join(
        config.DATASET_ROOT,
        "annotations",
        "trimaps",
        f"{image_id}.png",
    )


# ============================================================
# Read Oxford annotation files
# ============================================================

def read_split_file(split_file):
    """
    Oxford annotation format:

        image_id class_id species_id breed_id

    Example:

        Abyssinian_1 1 1 1
        american_bulldog_1 2 2 1

    species_id:
        1 -> Cat
        2 -> Dog

    Returns:
        image_ids: list
        species_map: dict
    """

    image_ids = []
    species_map = {}

    with open(
        split_file,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 3:
                continue

            image_id = parts[0]
            species_id = int(parts[2])

            image_ids.append(image_id)

            species_map[image_id] = species_id

    return image_ids, species_map


# ============================================================
# Validate Trimaps
# ============================================================

def inspect_trimaps(image_ids, verbose=True):
    """
    Filter invalid samples before training.

    A valid sample requires:

        1. Existing image.
        2. Existing Trimap.
        3. At least one foreground pixel.

    Returns:
        valid_ids
        invalid_ids
    """

    valid_ids = []
    invalid_ids = []

    for image_id in image_ids:

        image_path = get_image_path(image_id)
        trimap_path = get_trimap_path(image_id)

        if not os.path.isfile(image_path):

            invalid_ids.append(
                (image_id, "Image file not found")
            )

            continue

        if not os.path.isfile(trimap_path):

            invalid_ids.append(
                (image_id, "Trimap file not found")
            )

            continue

        try:

            with Image.open(trimap_path) as img:

                trimap = np.array(img)

            if trimap.ndim != 2:

                invalid_ids.append(
                    (image_id, "Trimap is not single-channel")
                )

                continue

            foreground = np.isin(
                trimap,
                FOREGROUND_VALUES,
            )

            if not np.any(foreground):

                invalid_ids.append(
                    (image_id, "Empty foreground mask")
                )

                continue

            valid_ids.append(image_id)

        except Exception as e:

            invalid_ids.append(
                (image_id, f"Failed to read Trimap: {e}")
            )

    if verbose:

        print()
        print("=" * 60)
        print("Trimap Validation")
        print("-" * 60)

        print(f"Input samples:   {len(image_ids)}")
        print(f"Valid samples:   {len(valid_ids)}")
        print(f"Invalid samples: {len(invalid_ids)}")

        if invalid_ids:

            print()
            print("Invalid samples:")

            for image_id, reason in invalid_ids:

                print(f"  {image_id}: {reason}")

        print("=" * 60)

    return valid_ids, invalid_ids


# ============================================================
# Generate mask and bounding box
# ============================================================

def generate_mask_and_box(trimap_path):
    """
    Generate binary mask and bounding box.

    Returns:
        mask: [H, W], uint8
        box: [xmin, ymin, xmax, ymax], float32
    """

    with Image.open(trimap_path) as img:

        trimap = np.array(img)

    mask = np.isin(
        trimap,
        FOREGROUND_VALUES,
    )

    ys, xs = np.where(mask)

    if len(xs) == 0:

        raise ValueError(
            f"Empty foreground mask: {trimap_path}"
        )

    xmin = float(xs.min())
    xmax = float(xs.max()) + 1.0

    ymin = float(ys.min())
    ymax = float(ys.max()) + 1.0

    box = np.array(
        [
            xmin,
            ymin,
            xmax,
            ymax,
        ],
        dtype=np.float32,
    )

    mask = mask.astype(np.uint8)

    return mask, box


# ============================================================
# Data augmentation
# ============================================================

def random_horizontal_flip(
    image,
    mask,
    box,
    probability=0.5,
):
    """
    Horizontally flip image, mask and bounding box.
    """

    if random.random() >= probability:

        return image, mask, box

    width, _ = image.size

    image = image.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    mask = np.fliplr(mask).copy()

    xmin, ymin, xmax, ymax = box

    new_xmin = width - xmax
    new_xmax = width - xmin

    box = np.array(
        [
            new_xmin,
            ymin,
            new_xmax,
            ymax,
        ],
        dtype=np.float32,
    )

    return image, mask, box


# ============================================================
# Oxford-IIIT Pet Dataset
# ============================================================

class OxfordPetDataset(Dataset):

    def __init__(
        self,
        image_ids,
        species_map,
        train=False,
    ):

        self.image_ids = list(image_ids)

        self.species_map = species_map

        self.train = train

        print(
            f"Dataset initialized: "
            f"{len(self.image_ids)} samples"
        )

    def __len__(self):

        return len(self.image_ids)

    def __getitem__(self, index):

        image_id = self.image_ids[index]

        image_path = get_image_path(image_id)

        trimap_path = get_trimap_path(image_id)

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        with Image.open(image_path) as img:

            image = img.convert("RGB")

        original_width, original_height = image.size

        # ----------------------------------------------------
        # Generate mask and bounding box
        # ----------------------------------------------------

        mask, box = generate_mask_and_box(
            trimap_path
        )

        if mask.shape != (
            original_height,
            original_width,
        ):

            raise ValueError(
                f"Image and Trimap size mismatch: {image_id}, "
                f"image={image.size}, "
                f"mask={mask.shape}"
            )

        # ----------------------------------------------------
        # Data augmentation
        # ----------------------------------------------------

        if self.train:

            image, mask, box = random_horizontal_flip(
                image,
                mask,
                box,
                probability=HORIZONTAL_FLIP_PROB,
            )

        # ----------------------------------------------------
        # Convert image to tensor
        # ----------------------------------------------------

        image_array = np.array(
            image,
            dtype=np.float32,
        ) / 255.0

        image_tensor = torch.from_numpy(
            image_array
        ).permute(
            2, 0, 1
        ).contiguous()

        # ----------------------------------------------------
        # Convert mask to tensor
        # ----------------------------------------------------

        mask_tensor = torch.from_numpy(
            mask
        ).to(
            dtype=torch.uint8
        )

        # One main pet instance per image.
        masks = mask_tensor.unsqueeze(0)

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        boxes = torch.from_numpy(
            box.reshape(1, 4)
        ).to(
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Class label
        # ----------------------------------------------------

        species_id = self.species_map[image_id]

        if species_id == 1:

            label = 1

        elif species_id == 2:

            label = 2

        else:

            raise ValueError(
                f"Unexpected species ID: "
                f"{species_id}, image: {image_id}"
            )

        labels = torch.tensor(
            [label],
            dtype=torch.int64,
        )

        # ----------------------------------------------------
        # Area
        # ----------------------------------------------------

        area = masks[0].sum().to(
            dtype=torch.float32
        ).unsqueeze(0)

        # ----------------------------------------------------
        # Iscrowd
        # ----------------------------------------------------

        iscrowd = torch.zeros(
            (1,),
            dtype=torch.int64,
        )

        # ----------------------------------------------------
        # Image ID
        # ----------------------------------------------------

        image_id_tensor = torch.tensor(
            [index],
            dtype=torch.int64,
        )

        # ----------------------------------------------------
        # Target dictionary
        # ----------------------------------------------------

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id_tensor,
            "area": area,
            "iscrowd": iscrowd,
        }

        return image_tensor, target


# ============================================================
# Collate function
# ============================================================

def collate_fn(batch):
    """
    Detection models require a custom collate function
    because images and targets can have different sizes.
    """

    return tuple(zip(*batch))


# ============================================================
# Build datasets
# ============================================================

def build_datasets():
    """
    Build train / validation / test datasets.

    Uses:
        trainval.txt -> train + validation
        test.txt     -> test

    Validation ratio:
        config.VAL_RATIO

    Invalid Trimaps are filtered before Dataset creation.
    """

    annotations_dir = os.path.join(
        config.DATASET_ROOT,
        "annotations",
    )

    trainval_file = os.path.join(
        annotations_dir,
        "trainval.txt",
    )

    test_file = os.path.join(
        annotations_dir,
        "test.txt",
    )

    # --------------------------------------------------------
    # Read split files
    # --------------------------------------------------------

    trainval_ids, trainval_species = read_split_file(
        trainval_file
    )

    test_ids, test_species = read_split_file(
        test_file
    )

    # --------------------------------------------------------
    # Shuffle trainval samples
    # --------------------------------------------------------

    rng = random.Random(
        config.SEED
    )

    rng.shuffle(
        trainval_ids
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    val_size = int(
        len(trainval_ids) * config.VAL_RATIO
    )

    val_ids = trainval_ids[:val_size]

    train_ids = trainval_ids[val_size:]

    # --------------------------------------------------------
    # Print original split sizes
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Oxford-IIIT Pet Dataset")
    print("-" * 60)

    print(f"Original Train: {len(train_ids)}")
    print(f"Original Val:   {len(val_ids)}")
    print(f"Original Test:  {len(test_ids)}")

    print("=" * 60)

    # --------------------------------------------------------
    # Validate training samples
    # --------------------------------------------------------

    print("\nChecking training samples...")

    train_ids, train_invalid = inspect_trimaps(
        train_ids,
        verbose=True,
    )

    # --------------------------------------------------------
    # Validate validation samples
    # --------------------------------------------------------

    print("\nChecking validation samples...")

    val_ids, val_invalid = inspect_trimaps(
        val_ids,
        verbose=True,
    )

    # --------------------------------------------------------
    # Validate test samples
    # --------------------------------------------------------

    print("\nChecking test samples...")

    test_ids, test_invalid = inspect_trimaps(
        test_ids,
        verbose=True,
    )

    # --------------------------------------------------------
    # Create Dataset objects
    # --------------------------------------------------------

    train_dataset = OxfordPetDataset(
        image_ids=train_ids,
        species_map=trainval_species,
        train=True,
    )

    val_dataset = OxfordPetDataset(
        image_ids=val_ids,
        species_map=trainval_species,
        train=False,
    )

    test_dataset = OxfordPetDataset(
        image_ids=test_ids,
        species_map=test_species,
        train=False,
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Final Dataset Statistics")
    print("-" * 60)

    print(
        f"Train: {len(train_dataset)} "
        f"(filtered {len(train_invalid)})"
    )

    print(
        f"Val:   {len(val_dataset)} "
        f"(filtered {len(val_invalid)})"
    )

    print(
        f"Test:  {len(test_dataset)} "
        f"(filtered {len(test_invalid)})"
    )

    print("=" * 60)
    print()

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# Compatibility alias
# ============================================================

def create_datasets():

    return build_datasets()


# ============================================================
# Create DataLoaders
# ============================================================

def create_dataloaders():

    train_dataset, val_dataset, test_dataset = build_datasets()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=getattr(config, "PIN_MEMORY", True),
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=getattr(config, "PIN_MEMORY", True),
        collate_fn=collate_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=getattr(config, "PIN_MEMORY", True),
        collate_fn=collate_fn,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )
