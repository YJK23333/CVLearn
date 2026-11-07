import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

import config

from dataset import (
    VOCDataset,
    collate_fn,
)

from model import FastRCNN

from utils import (
    load_checkpoint,
    postprocess_detections,
)


def compute_iou(box, boxes):

    if len(boxes) == 0:
        return np.zeros(
            (0,),
            dtype=np.float32,
        )

    x1 = np.maximum(
        box[0],
        boxes[:, 0],
    )

    y1 = np.maximum(
        box[1],
        boxes[:, 1],
    )

    x2 = np.minimum(
        box[2],
        boxes[:, 2],
    )

    y2 = np.minimum(
        box[3],
        boxes[:, 3],
    )

    inter = (
        np.maximum(0, x2 - x1 + 1)
        *
        np.maximum(0, y2 - y1 + 1)
    )

    area1 = (
        (box[2] - box[0] + 1)
        *
        (box[3] - box[1] + 1)
    )

    area2 = (
        (boxes[:, 2] - boxes[:, 0] + 1)
        *
        (boxes[:, 3] - boxes[:, 1] + 1)
    )

    union = area1 + area2 - inter

    return inter / np.maximum(
        union,
        1e-8,
    )


def compute_ap_11_point(recall, precision):

    ap = 0.0

    for threshold in np.arange(
        0.0,
        1.1,
        0.1,
    ):

        valid = recall >= threshold

        if np.any(valid):

            precision_value = np.max(
                precision[valid]
            )

        else:

            precision_value = 0.0

        ap += precision_value / 11.0

    return ap


@torch.no_grad()
def evaluate(model, dataloader):

    model.eval()

    all_detections = {
        class_id: []
        for class_id in range(
            1,
            config.NUM_CLASSES,
        )
    }

    all_ground_truths = {
        class_id: {}
        for class_id in range(
            1,
            config.NUM_CLASSES,
        )
    }

    for batch in tqdm(
        dataloader,
        desc="Testing",
    ):

        image = batch["image"].to(
            config.DEVICE
        )

        proposals = batch["proposals"].to(
            config.DEVICE
        )

        # Obtain resized image dimensions.
        # image shape: [3, H, W]
        image_size = (
            image.shape[-1],
            image.shape[-2],
        )

        outputs = model(
            [image],
            [proposals],
        )

        detections = postprocess_detections(
            outputs[0]["class_logits"],
            outputs[0]["bbox_deltas"],
            proposals,
            image_size,
        )

        image_id = batch["image_id"]

        for detection in detections:

            all_detections[
                detection["label"]
            ].append(
                {
                    "image_id": image_id,
                    "box": detection["box"].numpy(),
                    "score": detection["score"],
                }
            )

        gt_boxes = batch["gt_boxes"].numpy()
        gt_labels = batch["gt_labels"].numpy()
        gt_difficult = batch["difficult"].numpy().astype(bool)

        for class_id in range(
            1,
            config.NUM_CLASSES,
        ):

            mask = gt_labels == class_id

            all_ground_truths[
                class_id
            ][image_id] = {
                "boxes": gt_boxes[mask],
                "difficult": gt_difficult[mask],
            }

    ap_per_class = {}

    for class_id in range(
        1,
        config.NUM_CLASSES,
    ):

        detections = sorted(
            all_detections[class_id],
            key=lambda x: x["score"],
            reverse=True,
        )

        num_gt = 0

        for gt in all_ground_truths[class_id].values():

            num_gt += int(
                np.sum(~gt["difficult"])
            )

        if num_gt == 0:

            ap_per_class[class_id] = 0.0

            continue

        matched = {}

        for image_id, gt in all_ground_truths[class_id].items():

            matched[image_id] = np.zeros(
                len(gt["boxes"]),
                dtype=bool,
            )

        tp = []
        fp = []

        for detection in detections:

            image_id = detection["image_id"]

            gt = all_ground_truths[class_id][image_id]

            gt_boxes = gt["boxes"]
            gt_difficult = gt["difficult"]

            if len(gt_boxes) == 0:

                tp.append(0)
                fp.append(1)

                continue

            ious = compute_iou(
                detection["box"],
                gt_boxes,
            )

            best_idx = int(
                np.argmax(ious)
            )

            best_iou = ious[best_idx]

            if best_iou < config.IOU_THRESHOLD:

                tp.append(0)
                fp.append(1)

                continue

            # Ignore detections matching difficult objects.
            if gt_difficult[best_idx]:
                continue

            if not matched[image_id][best_idx]:

                tp.append(1)
                fp.append(0)

                matched[image_id][best_idx] = True

            else:

                tp.append(0)
                fp.append(1)

        tp = np.asarray(
            tp,
            dtype=np.float32,
        )

        fp = np.asarray(
            fp,
            dtype=np.float32,
        )

        if len(tp) == 0:

            ap_per_class[class_id] = 0.0

            continue

        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)

        recall = cumulative_tp / max(
            num_gt,
            1,
        )

        precision = cumulative_tp / np.maximum(
            cumulative_tp + cumulative_fp,
            1e-8,
        )

        ap = compute_ap_11_point(
            recall,
            precision,
        )

        ap_per_class[class_id] = ap

    return ap_per_class


def main():

    print("=" * 60)
    print("Fast R-CNN Evaluation")
    print("=" * 60)

    dataset = VOCDataset(
        split="test"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    model = FastRCNN().to(
        config.DEVICE
    )

    checkpoint_path = (
        config.CHECKPOINT_DIR
        / config.BEST_CHECKPOINT_NAME
    )

    load_checkpoint(
        model,
        path=checkpoint_path,
    )

    ap_per_class = evaluate(
        model,
        dataloader,
    )

    print("\nAP per class:")

    ap_values = []

    for class_id in range(
        1,
        config.NUM_CLASSES,
    ):

        ap = ap_per_class[class_id]

        ap_values.append(ap)

        print(
            f"{config.VOC_CLASSES[class_id]:15s}: {ap:.4f}"
        )

    map_value = np.mean(
        ap_values
    )

    print("\n" + "=" * 60)

    print(
        f"VOC2007 mAP@0.5: {map_value:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()