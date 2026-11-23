
import random
import xml.etree.ElementTree as ET

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision.transforms import functional as TF

import config


class VOCDataset(Dataset):

    def __init__(
        self,
        root,
        split="train",
        min_size=config.MIN_SIZE,
        max_size=config.MAX_SIZE,
        augment=False,
    ):

        self.root = Path(root)
        self.split = split

        self.min_size = min_size
        self.max_size = max_size

        self.augment = augment

        split_file = (
            self.root
            / "ImageSets"
            / "Main"
            / f"{split}.txt"
        )

        with open(split_file, "r") as f:
            self.image_ids = [
                line.strip()
                for line in f
                if line.strip()
            ]

    def __len__(self):

        return len(self.image_ids)

    def _load_annotation(self, image_id):

        xml_path = (
            self.root
            / "Annotations"
            / f"{image_id}.xml"
        )

        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):

            name = obj.find("name").text

            if name not in config.CLASS_TO_IDX:
                continue

            difficult = int(
                obj.findtext("difficult", default="0")
            )

            if difficult:
                continue

            bbox = obj.find("bndbox")

            x1 = float(bbox.findtext("xmin"))
            y1 = float(bbox.findtext("ymin"))
            x2 = float(bbox.findtext("xmax"))
            y2 = float(bbox.findtext("ymax"))

            # VOC coordinates are 1-based.
            # Convert to zero-based continuous coordinates.
            boxes.append([
                x1 - 1,
                y1 - 1,
                x2 - 1,
                y2 - 1,
            ])

            labels.append(
                config.CLASS_TO_IDX[name]
            )

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        ).reshape(-1, 4)

        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        return boxes, labels

    def _resize(self, image, boxes):

        width, height = image.size

        scale = self.min_size / min(width, height)

        if max(width, height) * scale > self.max_size:

            scale = self.max_size / max(width, height)

        new_width = round(width * scale)
        new_height = round(height * scale)

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR
        )

        scale_x = new_width / width
        scale_y = new_height / height

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
            0, new_width
        )

        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
            0, new_height
        )

        return image, boxes

    def __getitem__(self, index):

        image_id = self.image_ids[index]

        image_path = (
            self.root
            / "JPEGImages"
            / f"{image_id}.jpg"
        )

        image = Image.open(image_path).convert("RGB")

        boxes, labels = self._load_annotation(
            image_id
        )

        image, boxes = self._resize(
            image,
            boxes
        )

        if self.augment and random.random() < 0.5:

            image = TF.hflip(image)

            width, _ = image.size

            old_x1 = boxes[:, 0].clone()
            old_x2 = boxes[:, 2].clone()

            boxes[:, 0] = width - old_x2
            boxes[:, 2] = width - old_x1

        image = TF.to_tensor(image)

        image = TF.normalize(
            image,
            mean=config.IMAGE_MEAN,
            std=config.IMAGE_STD
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
        }

        return image, target


def collate_fn(batch):

    images, targets = zip(*batch)

    return list(images), list(targets)


def get_dataloaders():

    train_dataset = VOCDataset(
        root=config.DATA_ROOT,
        split="train",
        augment=True
    )

    val_dataset = VOCDataset(
        root=config.DATA_ROOT,
        split="val",
        augment=False
    )

    test_dataset = VOCDataset(
        root=config.DATA_ROOT,
        split="test",
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
