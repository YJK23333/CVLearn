
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

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

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

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_images = 0

    total_ground_truth = 0

    true_positive = 0

    false_positive = 0

    false_negative = 0

    # --------------------------------------------------------
    # IoU Records
    # --------------------------------------------------------

    mask_ious = []

    box_ious = []

    # --------------------------------------------------------
    # Visualization Directory
    # --------------------------------------------------------

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

    # ========================================================
    # Iterate Dataset
    # ========================================================

    for index, (images, targets) in enumerate(progress):

        images = [
            image.to(device)
            for image in images
        ]

        predictions = model(images)

        prediction = predictions[0]

        target = targets[0]

        # ----------------------------------------------------
        # Ground Truth
        # ----------------------------------------------------

        gt_mask = target["masks"][0].bool()

        gt_box = target["boxes"][0].cpu().numpy()

        gt_label = int(
            target["labels"][0]
        )

        total_images += 1

        total_ground_truth += 1

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        pred_scores = prediction["scores"].cpu()

        pred_labels = prediction["labels"].cpu()

        pred_boxes = prediction["boxes"].cpu()

        pred_masks = prediction["masks"].cpu()

        # ----------------------------------------------------
        # Filter Predictions
        # ----------------------------------------------------

        valid_indices = [

            i

            for i in range(len(pred_scores))

            if (
                float(pred_scores[i]) >= config.SCORE_THRESHOLD
                and int(pred_labels[i]) > 0
            )
        ]

        # ----------------------------------------------------
        # Match Predictions with Ground Truth
        # ----------------------------------------------------

        candidates = []

        for pred_index in valid_indices:

            # Class must match.

            if int(pred_labels[pred_index]) != gt_label:

                continue

            pred_mask = (
                pred_masks[pred_index, 0]
                >= config.MASK_THRESHOLD
            )

            # Mask IoU

            mask_iou = calculate_iou(
                gt_mask,
                pred_mask
            )

            # Box IoU

            box_iou = calculate_box_iou(
                gt_box,
                pred_boxes[pred_index].numpy()
            )

            candidates.append({

                "index": pred_index,

                "score": float(
                    pred_scores[pred_index]
                ),

                "mask_iou": mask_iou,

                "box_iou": box_iou
            })

        # ----------------------------------------------------
        # Best-Match IoU
        # ----------------------------------------------------

        if len(candidates) > 0:

            best_candidate = max(
                candidates,
                key=lambda x: x["mask_iou"]
            )

            best_mask_iou = best_candidate["mask_iou"]

            best_box_iou = best_candidate["box_iou"]

        else:

            best_mask_iou = 0.0

            best_box_iou = 0.0

        mask_ious.append(
            best_mask_iou
        )

        box_ious.append(
            best_box_iou
        )

        # ----------------------------------------------------
        # Detection Matching
        # ----------------------------------------------------

        matched_candidates = [

            candidate

            for candidate in candidates

            if (
                candidate["mask_iou"]
                >= config.IOU_THRESHOLD
            )
        ]

        if len(matched_candidates) > 0:

            # Select the highest-confidence
            # prediction that satisfies the IoU threshold.

            matched_candidate = max(
                matched_candidates,
                key=lambda x: x["score"]
            )

            true_positive += 1

            # Every other valid prediction is a false positive.

            false_positive += (
                len(valid_indices) - 1
            )

        else:

            # No valid prediction matches the Ground Truth.

            false_negative += 1

            # All valid predictions are false positives.

            false_positive += len(valid_indices)

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

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        progress.set_postfix(

            mask_iou=f"{np.mean(mask_ious):.4f}",

            TP=true_positive,

            FP=false_positive,

            FN=false_negative
        )

    # ========================================================
    # 3. Final Metrics
    # ========================================================

    precision = (
        true_positive /
        max(true_positive + false_positive, 1)
    )

    recall = (
        true_positive /
        max(total_ground_truth, 1)
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

    # ========================================================
    # 4. Print Results
    # ========================================================

    print("\n" + "=" * 60)

    print("Mask R-CNN Test Results")

    print("=" * 60)

    print(f"Test Images: {total_images}")

    print(f"Ground Truth Instances: {total_ground_truth}")

    print("-" * 60)

    print(f"True Positive: {true_positive}")

    print(f"False Positive: {false_positive}")

    print(f"False Negative: {false_negative}")

    print("-" * 60)

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
# 5. Main
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

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate(
        model,
        test_loader,
        device
    )


if __name__ == "__main__":

    main()
