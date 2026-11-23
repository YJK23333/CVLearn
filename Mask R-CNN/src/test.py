
import os

import torch

import numpy as np

from torch.utils.data import DataLoader

from tqdm import tqdm

import config

from dataset import (
    build_datasets,
    collate_fn
)

from model import build_model

from utils import get_device

from visualize import save_visualization


# ============================================================
# 1. IoU
# ============================================================

def calculate_iou(mask1, mask2):

    mask1 = mask1.bool()

    mask2 = mask2.bool()

    intersection = (
        mask1 & mask2
    ).sum().item()

    union = (
        mask1 | mask2
    ).sum().item()

    if union == 0:

        return 0.0

    return intersection / union


def calculate_box_iou(box1, box2):

    x1 = max(box1[0], box2[0])

    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])

    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:

        return 0.0

    return intersection / union


# ============================================================
# 2. Evaluate
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):

    model.eval()

    total_images = 0

    true_positive = 0

    false_positive = 0

    false_negative = 0

    mask_ious = []

    box_ious = []

    visualization_dir = os.path.join(
        config.RESULT_DIR,
        "visualizations"
    )

    os.makedirs(
        visualization_dir,
        exist_ok=True
    )

    progress = tqdm(
        loader,
        desc="Testing"
    )

    for index, (images, targets) in enumerate(progress):

        images = [
            image.to(device)
            for image in images
        ]

        predictions = model(images)

        prediction = predictions[0]

        target = targets[0]

        gt_mask = target["masks"][0].bool()

        gt_box = target["boxes"][0].cpu().numpy()

        gt_label = int(
            target["labels"][0]
        )

        pred_scores = prediction["scores"].cpu()

        pred_labels = prediction["labels"].cpu()

        pred_boxes = prediction["boxes"].cpu()

        pred_masks = prediction["masks"].cpu()

        valid_indices = [

            i

            for i in range(len(pred_scores))

            if (
                float(pred_scores[i]) >= config.SCORE_THRESHOLD
                and int(pred_labels[i]) > 0
            )
        ]

        total_images += 1

        best_iou = 0.0

        best_box_iou = 0.0

        best_index = None

        # ----------------------------------------------------
        # Match prediction with ground truth
        # ----------------------------------------------------

        for pred_index in valid_indices:

            if int(pred_labels[pred_index]) != gt_label:

                continue

            pred_mask = (
                pred_masks[pred_index, 0]
                >= config.MASK_THRESHOLD
            )

            mask_iou = calculate_iou(
                gt_mask,
                pred_mask
            )

            if mask_iou > best_iou:

                best_iou = mask_iou

                best_index = pred_index

                best_box_iou = calculate_box_iou(
                    gt_box,
                    pred_boxes[pred_index].numpy()
                )

        mask_ious.append(best_iou)

        box_ious.append(best_box_iou)

        # ----------------------------------------------------
        # Detection Metrics
        # ----------------------------------------------------

        matched = (
            best_index is not None
            and best_iou >= config.IOU_THRESHOLD
        )

        if matched:

            true_positive += 1

            false_positive += (
                len(valid_indices) - 1
            )

        else:

            false_positive += len(valid_indices)

            false_negative += 1

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        if index < config.MAX_VISUALIZE_IMAGES:

            save_path = os.path.join(
                visualization_dir,
                f"result_{index:04d}.jpg"
            )

            save_visualization(
                images[0].cpu(),
                target,
                prediction,
                save_path
            )

        progress.set_postfix(
            mask_iou=f"{np.mean(mask_ious):.4f}"
        )

    # --------------------------------------------------------
    # Final Metrics
    # --------------------------------------------------------

    precision = (
        true_positive /
        max(true_positive + false_positive, 1)
    )

    recall = (
        true_positive /
        max(true_positive + false_negative, 1)
    )

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-8)
    )

    mean_mask_iou = float(
        np.mean(mask_ious)
    )

    mean_box_iou = float(
        np.mean(box_ious)
    )

    print("\n" + "=" * 60)

    print("Mask R-CNN Test Results")

    print("=" * 60)

    print(f"Test Images: {total_images}")

    print(f"Mask IoU: {mean_mask_iou:.4f}")

    print(f"Box IoU: {mean_box_iou:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall: {recall:.4f}")

    print(f"F1 Score: {f1:.4f}")

    print("=" * 60)

    print(
        f"Visualizations saved to: {visualization_dir}"
    )


# ============================================================
# 3. Main
# ============================================================

def main():

    config.create_directories()

    device = get_device()

    (
        _,
        _,
        test_dataset
    ) = build_datasets()

    test_loader = DataLoader(

        test_dataset,

        batch_size=1,

        shuffle=False,

        num_workers=config.NUM_WORKERS,

        pin_memory=(
            config.PIN_MEMORY
            and device.type == "cuda"
        ),

        collate_fn=collate_fn
    )

    # --------------------------------------------------------
    # Load Model
    # --------------------------------------------------------

    model = build_model()

    model.to(device)

    checkpoint_path = config.BEST_CHECKPOINT

    if not os.path.exists(checkpoint_path):

        checkpoint_path = config.LAST_CHECKPOINT

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            "No trained checkpoint found."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Loaded checkpoint: {checkpoint_path}"
    )

    print(
        f"Checkpoint epoch: {checkpoint['epoch']}"
    )

    evaluate(
        model,
        test_loader,
        device
    )


if __name__ == "__main__":

    main()
