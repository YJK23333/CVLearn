
import random

import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

import config

from dataset import VOCDataset
from model import FastRCNN


# ============================================================
# Visualization Settings
# ============================================================

NUM_IMAGES = 10

RANDOM_SEED = None

OUTPUT_DIR = config.OUTPUT_DIR / "visualizations"

CHECKPOINT_PATH = (
    config.CHECKPOINT_DIR
    / config.BEST_CHECKPOINT_NAME
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Bounding Box Decoding
# ============================================================

def decode_boxes(
    proposals,
    deltas,
):
    """
    Decode bounding box regression deltas.

    proposals: [N, 4]
    deltas:    [N, 4]

    Returns:
        decoded_boxes: [N, 4]
    """

    pw = proposals[:, 2] - proposals[:, 0] + 1
    ph = proposals[:, 3] - proposals[:, 1] + 1

    px = proposals[:, 0] + 0.5 * pw
    py = proposals[:, 1] + 0.5 * ph

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    gx = dx * pw + px
    gy = dy * ph + py

    gw = torch.exp(dw) * pw
    gh = torch.exp(dh) * ph

    x1 = gx - 0.5 * gw
    y1 = gy - 0.5 * gh

    x2 = gx + 0.5 * gw - 1
    y2 = gy + 0.5 * gh - 1

    decoded_boxes = torch.stack(
        [x1, y1, x2, y2],
        dim=1,
    )

    return decoded_boxes


def clip_boxes(
    boxes,
    image_width,
    image_height,
):
    """
    Clip boxes to image boundaries.
    """

    boxes[:, 0].clamp_(0, image_width - 1)
    boxes[:, 1].clamp_(0, image_height - 1)

    boxes[:, 2].clamp_(0, image_width - 1)
    boxes[:, 3].clamp_(0, image_height - 1)

    return boxes


# ============================================================
# Detection Postprocessing
# ============================================================

def predict_detections(
    output,
    proposals,
    image_size,
):
    """
    Convert model outputs into final detections.

    Returns:
        List of dictionaries:
        {
            "boxes":  [N, 4],
            "scores": [N],
            "labels": [N],
        }
    """

    image_width, image_height = image_size

    class_logits = output["class_logits"]

    bbox_deltas = output["bbox_deltas"]

    probabilities = F.softmax(
        class_logits,
        dim=1,
    )

    bbox_deltas = bbox_deltas.reshape(
        -1,
        config.BBOX_REG_CLASSES,
        4,
    )

    all_boxes = []
    all_scores = []
    all_labels = []

    # Skip background class 0.
    for class_id in range(1, config.NUM_CLASSES):

        scores = probabilities[:, class_id]

        # Confidence filtering.
        keep = scores >= config.CONFIDENCE_THRESHOLD

        if not keep.any():
            continue

        class_scores = scores[keep]

        class_proposals = proposals[keep]

        class_deltas = bbox_deltas[
            keep,
            class_id - 1,
        ]

        # Decode bounding boxes.
        boxes = decode_boxes(
            class_proposals,
            class_deltas,
        )

        boxes = clip_boxes(
            boxes,
            image_width,
            image_height,
        )

        # Remove invalid boxes.
        valid = (
            (boxes[:, 2] >= boxes[:, 0])
            &
            (boxes[:, 3] >= boxes[:, 1])
        )

        boxes = boxes[valid]
        class_scores = class_scores[valid]

        if len(boxes) == 0:
            continue

        # torchvision NMS uses continuous coordinates.
        # Convert inclusive boxes to continuous coordinates.
        nms_boxes = boxes.clone()

        nms_boxes[:, 2] += 1
        nms_boxes[:, 3] += 1

        keep_indices = nms(
            nms_boxes,
            class_scores,
            config.NMS_THRESHOLD,
        )

        boxes = boxes[keep_indices]
        class_scores = class_scores[keep_indices]

        labels = torch.full(
            (len(boxes),),
            class_id,
            dtype=torch.long,
            device=boxes.device,
        )

        all_boxes.append(boxes)
        all_scores.append(class_scores)
        all_labels.append(labels)

    if not all_boxes:

        return {
            "boxes": torch.empty((0, 4)),
            "scores": torch.empty((0,)),
            "labels": torch.empty((0,), dtype=torch.long),
        }

    boxes = torch.cat(all_boxes, dim=0)

    scores = torch.cat(all_scores, dim=0)

    labels = torch.cat(all_labels, dim=0)

    # Global confidence sorting.
    order = torch.argsort(
        scores,
        descending=True,
    )

    order = order[
        :config.MAX_DETECTIONS_PER_IMAGE
    ]

    return {
        "boxes": boxes[order].detach().cpu(),
        "scores": scores[order].detach().cpu(),
        "labels": labels[order].detach().cpu(),
    }


# ============================================================
# Image Processing
# ============================================================

def denormalize_image(image_tensor):
    """
    Convert normalized tensor back to PIL image.
    """

    mean = torch.tensor(
        config.IMAGE_MEAN,
        dtype=image_tensor.dtype,
    ).view(3, 1, 1)

    std = torch.tensor(
        config.IMAGE_STD,
        dtype=image_tensor.dtype,
    ).view(3, 1, 1)

    image = image_tensor.cpu() * std + mean

    image = image.clamp(0, 1)

    image = (
        image.permute(1, 2, 0)
        .numpy()
        * 255
    ).astype(np.uint8)

    return Image.fromarray(image)


# ============================================================
# Drawing
# ============================================================

def draw_box(
    draw,
    box,
    color,
    text,
    font,
    width=3,
):
    """
    Draw one bounding box and its label.
    """

    x1, y1, x2, y2 = [
        int(round(value))
        for value in box
    ]

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=color,
        width=width,
    )

    text_bbox = draw.textbbox(
        (0, 0),
        text,
        font=font,
    )

    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    text_y = max(
        0,
        y1 - text_height - 6,
    )

    draw.rectangle(
        [
            x1,
            text_y,
            x1 + text_width + 6,
            text_y + text_height + 4,
        ],
        fill=color,
    )

    draw.text(
        (x1 + 3, text_y + 1),
        text,
        fill="white",
        font=font,
    )


def visualize_image(
    image,
    detections,
    gt_boxes,
    gt_labels,
    difficult,
    image_id,
):
    """
    Draw predictions and ground truth.
    """

    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()

    # Draw ground truth first.
    for box, label, is_difficult in zip(
        gt_boxes,
        gt_labels,
        difficult,
    ):

        class_name = config.VOC_CLASSES[
            int(label)
        ]

        suffix = " difficult" if bool(is_difficult) else ""

        text = f"GT: {class_name}{suffix}"

        draw_box(
            draw,
            box.tolist(),
            color="green",
            text=text,
            font=font,
            width=3,
        )

    # Draw model predictions.
    for box, score, label in zip(
        detections["boxes"],
        detections["scores"],
        detections["labels"],
    ):

        class_name = config.VOC_CLASSES[
            int(label)
        ]

        text = (
            f"{class_name} "
            f"{float(score):.2f}"
        )

        draw_box(
            draw,
            box.tolist(),
            color="red",
            text=text,
            font=font,
            width=2,
        )

    return image


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Fast R-CNN Visualization")
    print("=" * 60)

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = VOCDataset(
        split="test"
    )

    print(f"Test images: {len(dataset)}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = FastRCNN().to(
        config.DEVICE
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=config.DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {config.DEVICE}")

    # --------------------------------------------------------
    # Select random images
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    num_images = min(
        NUM_IMAGES,
        len(dataset),
    )

    selected_indices = random.sample(
        range(len(dataset)),
        num_images,
    )

    print(
        f"\nSelected {num_images} images:"
    )

    # --------------------------------------------------------
    # Inference and visualization
    # --------------------------------------------------------

    for index in selected_indices:

        batch = dataset[index]

        image = batch["image"]

        proposals = batch["proposals"]

        image_id = batch["image_id"]

        gt_boxes = batch["gt_boxes"]

        gt_labels = batch["gt_labels"]

        difficult = batch["difficult"]

        image_height = image.shape[1]
        image_width = image.shape[2]

        image_size = (
            image_width,
            image_height,
        )

        # Move tensors to GPU.
        image_device = image.unsqueeze(0).to(
            config.DEVICE
        )

        proposals_device = proposals.to(
            config.DEVICE
        )

        # Model inference.
        with torch.no_grad():

            with torch.amp.autocast(
                device_type=config.DEVICE.type,
                enabled=config.AMP,
            ):

                outputs = model(
                    [image_device[0]],
                    [proposals_device],
                )

            detections = predict_detections(
                outputs[0],
                proposals_device,
                image_size,
            )

        # Convert normalized image back to PIL.
        visual_image = denormalize_image(
            image
        )

        # Draw boxes.
        visual_image = visualize_image(
            visual_image,
            detections,
            gt_boxes,
            gt_labels,
            difficult,
            image_id,
        )

        # Save image.
        save_path = (
            OUTPUT_DIR
            / f"{image_id}.jpg"
        )

        visual_image.save(
            save_path,
            quality=95,
        )

        print(
            f"[{selected_indices.index(index) + 1}/{num_images}] "
            f"{image_id}: "
            f"{len(detections['boxes'])} detections "
            f"-> {save_path}"
        )

    print("\nVisualization completed.")

    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":

    main()
