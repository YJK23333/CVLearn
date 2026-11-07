
import random
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

import config
from selective_search import load_or_generate_proposals


class VOCDataset(Dataset):
    """
    PASCAL VOC 2007 Dataset for Fast R-CNN.

    Train / Val:
        Returns sampled proposals, labels and bbox regression targets.

    Test:
        Returns all proposals and ground-truth annotations.
    """

    def __init__(self, split="train"):
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Unsupported dataset split: {split}"
            )

        self.split = split
        self.voc_root = config.VOC_ROOT

        split_file = (
            self.voc_root
            / "ImageSets"
            / "Main"
            / f"{split}.txt"
        )

        with open(split_file, "r", encoding="utf-8") as f:
            self.image_ids = [
                line.strip().split()[0]
                for line in f
                if line.strip()
            ]

        self.image_dir = self.voc_root / "JPEGImages"
        self.annotation_dir = self.voc_root / "Annotations"

        self.is_train = split in ("train", "val")

    def __len__(self):
        return len(self.image_ids)

    def _load_annotation(self, image_id):
        """
        Load VOC XML annotations.

        Returns:
            boxes: [N, 4], xyxy format.
            labels: [N]
            difficult: [N]
        """
        xml_path = self.annotation_dir / f"{image_id}.xml"
        root = ET.parse(xml_path).getroot()

        boxes = []
        labels = []
        difficult = []

        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()

            if name not in config.CLASS_TO_ID:
                continue

            bbox = obj.find("bndbox")

            # VOC coordinates are 1-based.
            # Convert to zero-based inclusive coordinates.
            x1 = float(bbox.find("xmin").text) - 1
            y1 = float(bbox.find("ymin").text) - 1
            x2 = float(bbox.find("xmax").text) - 1
            y2 = float(bbox.find("ymax").text) - 1

            boxes.append([x1, y1, x2, y2])
            labels.append(config.CLASS_TO_ID[name])

            difficult.append(
                int(obj.findtext("difficult", default="0"))
            )

        if boxes:
            boxes = torch.tensor(
                boxes,
                dtype=torch.float32
            )
            labels = torch.tensor(
                labels,
                dtype=torch.long
            )
            difficult = torch.tensor(
                difficult,
                dtype=torch.bool
            )
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            difficult = torch.zeros((0,), dtype=torch.bool)

        return boxes, labels, difficult

    def _resize_image(self, image):
        """
        Resize image while preserving aspect ratio.

        The shorter side is IMAGE_MIN_SIZE.
        The longer side does not exceed IMAGE_MAX_SIZE.
        """
        original_width, original_height = image.size

        min_size = config.IMAGE_MIN_SIZE
        max_size = config.IMAGE_MAX_SIZE

        scale = min_size / min(
            original_width,
            original_height
        )

        if max(original_width, original_height) * scale > max_size:
            scale = max_size / max(
                original_width,
                original_height
            )

        new_width = int(round(original_width * scale))
        new_height = int(round(original_height * scale))

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR
        )

        # Use the actual rounded dimensions to calculate scale.
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        return (
            image,
            scale_x,
            scale_y,
            new_width,
            new_height
        )

    @staticmethod
    def _scale_boxes(
        boxes,
        scale_x,
        scale_y,
        width,
        height
    ):
        """
        Scale boxes and clip them to resized image boundaries.

        Box format: [x1, y1, x2, y2].
        """
        boxes = boxes.clone()

        if boxes.numel() == 0:
            return boxes

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
            0,
            width - 1
        )

        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
            0,
            height - 1
        )

        return boxes

    @staticmethod
    def _calculate_iou(boxes1, boxes2):
        """
        Calculate IoU using inclusive xyxy coordinates.

        This is consistent with the +1 convention used
        in the bounding-box regression calculations.
        """
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros(
                (boxes1.shape[0], boxes2.shape[0]),
                dtype=torch.float32
            )

        top_left = torch.maximum(
            boxes1[:, None, :2],
            boxes2[None, :, :2]
        )

        bottom_right = torch.minimum(
            boxes1[:, None, 2:],
            boxes2[None, :, 2:]
        )

        intersection_wh = (
            bottom_right - top_left + 1
        ).clamp(min=0)

        intersection = (
            intersection_wh[..., 0]
            * intersection_wh[..., 1]
        )

        area1 = (
            (boxes1[:, 2] - boxes1[:, 0] + 1).clamp(min=0)
            * (boxes1[:, 3] - boxes1[:, 1] + 1).clamp(min=0)
        )

        area2 = (
            (boxes2[:, 2] - boxes2[:, 0] + 1).clamp(min=0)
            * (boxes2[:, 3] - boxes2[:, 1] + 1).clamp(min=0)
        )

        union = (
            area1[:, None]
            + area2[None, :]
            - intersection
        )

        return intersection / union.clamp(min=1e-6)

    @staticmethod
    def _encode_boxes(proposals, gt_boxes):
        """
        Encode ground-truth boxes relative to proposals.

        Returns:
            [dx, dy, dw, dh]
        """
        pw = proposals[:, 2] - proposals[:, 0] + 1
        ph = proposals[:, 3] - proposals[:, 1] + 1

        px = proposals[:, 0] + 0.5 * pw
        py = proposals[:, 1] + 0.5 * ph

        gw = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
        gh = gt_boxes[:, 3] - gt_boxes[:, 1] + 1

        gx = gt_boxes[:, 0] + 0.5 * gw
        gy = gt_boxes[:, 1] + 0.5 * gh

        dx = (gx - px) / pw
        dy = (gy - py) / ph

        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)

        return torch.stack(
            [dx, dy, dw, dh],
            dim=1
        )

    def _assign_targets(
        self,
        proposals,
        gt_boxes,
        gt_labels,
        index
    ):
        """
        Assign classification labels and bbox targets to proposals.

        Positive:
            IoU >= POSITIVE_IOU_THRESHOLD

        Negative:
            IoU < NEGATIVE_IOU_THRESHOLD

        Other proposals are ignored.
        """
        num_proposals = proposals.shape[0]

        labels = torch.full(
            (num_proposals,),
            -1,
            dtype=torch.long
        )

        bbox_targets = torch.zeros(
            (num_proposals, 4),
            dtype=torch.float32
        )

        if num_proposals == 0:
            raise RuntimeError(
                "No proposals available for this image."
            )

        if gt_boxes.shape[0] == 0:
            labels[:] = 0

        else:
            ious = self._calculate_iou(
                proposals,
                gt_boxes
            )

            max_iou, matched_indices = ious.max(dim=1)

            matched_labels = gt_labels[matched_indices]

            positive = (
                max_iou >= config.POSITIVE_IOU_THRESHOLD
            )

            negative = (
                max_iou < config.NEGATIVE_IOU_THRESHOLD
            )

            labels[positive] = matched_labels[positive]
            labels[negative] = 0

            if positive.any():
                matched_gt = gt_boxes[
                    matched_indices[positive]
                ]

                bbox_targets[positive] = self._encode_boxes(
                    proposals[positive],
                    matched_gt
                )

        if not self.is_train:
            return proposals, labels, bbox_targets

        # Sample positive and negative proposals.
        positive_indices = torch.where(labels > 0)[0].tolist()
        negative_indices = torch.where(labels == 0)[0].tolist()

        rng = random.Random(
            config.RANDOM_SEED + index
        )

        max_positive = config.MAX_POSITIVE_ROIS
        max_negative = config.MAX_NEGATIVE_ROIS
        max_total = config.MAX_ROIS_PER_IMAGE

        if len(positive_indices) > max_positive:
            positive_indices = rng.sample(
                positive_indices,
                max_positive
            )

        remaining_negative = min(
            max_negative,
            max_total - len(positive_indices)
        )

        if len(negative_indices) > remaining_negative:
            negative_indices = rng.sample(
                negative_indices,
                remaining_negative
            )

        keep_indices = (
            positive_indices + negative_indices
        )

        if not keep_indices:
            # Fallback to one background proposal.
            keep_indices = [0]
            labels[0] = 0

        keep_indices = torch.tensor(
            keep_indices,
            dtype=torch.long
        )

        return (
            proposals[keep_indices],
            labels[keep_indices],
            bbox_targets[keep_indices]
        )

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_path = (
            self.image_dir / f"{image_id}.jpg"
        )

        image = Image.open(image_path).convert("RGB")

        original_width, original_height = image.size

        # ----------------------------------------------------
        # 1. Generate or load proposals in original coordinates
        # ----------------------------------------------------

        proposals = load_or_generate_proposals(
            image_path
        )

        proposals = torch.as_tensor(
            proposals,
            dtype=torch.float32
        ).reshape(-1, 4)

        # ----------------------------------------------------
        # 2. Load ground-truth annotations
        # ----------------------------------------------------

        gt_boxes, gt_labels, difficult = (
            self._load_annotation(image_id)
        )

        # ----------------------------------------------------
        # 3. Resize image
        # ----------------------------------------------------

        (
            image,
            scale_x,
            scale_y,
            new_width,
            new_height
        ) = self._resize_image(image)

        # ----------------------------------------------------
        # 4. Scale proposals and GT boxes
        #
        # IMPORTANT:
        # This operation applies to train, val and test.
        # ----------------------------------------------------

        proposals = self._scale_boxes(
            proposals,
            scale_x,
            scale_y,
            new_width,
            new_height
        )

        gt_boxes = self._scale_boxes(
            gt_boxes,
            scale_x,
            scale_y,
            new_width,
            new_height
        )

        # ----------------------------------------------------
        # 5. Convert image to normalized tensor
        # ----------------------------------------------------

        image = TF.to_tensor(image)

        image = TF.normalize(
            image,
            mean=config.IMAGE_MEAN,
            std=config.IMAGE_STD
        )

        # ----------------------------------------------------
        # 6. Train / Validation
        # ----------------------------------------------------

        if self.is_train:
            proposals, labels, bbox_targets = (
                self._assign_targets(
                    proposals,
                    gt_boxes,
                    gt_labels,
                    index
                )
            )

            return {
                "image": image,
                "proposals": proposals,
                "labels": labels,
                "bbox_targets": bbox_targets,
                "image_id": image_id
            }

        # ----------------------------------------------------
        # 7. Test
        # ----------------------------------------------------

        return {
            "image": image,
            "proposals": proposals,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "difficult": difficult,
            "image_id": image_id
        }


def collate_fn(batch):
    """
    Fast R-CNN processes one image and its RoIs per iteration.
    """
    return batch[0]
