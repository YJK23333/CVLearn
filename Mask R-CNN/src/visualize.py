
import os

import numpy as np

import torch

from PIL import Image, ImageDraw

import config


# ============================================================
# 1. Colors
# ============================================================

CLASS_COLORS = {

    1: (255, 60, 60),

    2: (50, 150, 255)
}


# ============================================================
# 2. Draw Mask
# ============================================================

def draw_mask(image, mask, color, alpha=0.45):

    image_array = np.array(
        image
    ).astype(np.float32)

    mask = np.asarray(mask).astype(bool)

    color_array = np.array(
        color,
        dtype=np.float32
    )

    image_array[mask] = (
        image_array[mask] * (1 - alpha)
        + color_array * alpha
    )

    image_array = np.clip(
        image_array,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(image_array)


# ============================================================
# 3. Draw Ground Truth
# ============================================================

def draw_ground_truth(image, target):

    result = image.copy()

    masks = target["masks"]

    boxes = target["boxes"]

    labels = target["labels"]

    for i in range(len(labels)):

        label = int(labels[i])

        color = CLASS_COLORS[label]

        mask = masks[i].cpu().numpy()

        result = draw_mask(
            result,
            mask,
            color
        )

        draw = ImageDraw.Draw(result)

        box = boxes[i].cpu().numpy().tolist()

        draw.rectangle(
            box,
            outline=color,
            width=3
        )

        draw.text(
            (box[0], max(0, box[1] - 15)),
            f"GT: {config.CLASS_NAMES[label]}",
            fill=color
        )

    return result


# ============================================================
# 4. Draw Predictions
# ============================================================

def draw_predictions(
    image,
    prediction,
    score_threshold=config.SCORE_THRESHOLD
):

    result = image.copy()

    boxes = prediction["boxes"].cpu()

    labels = prediction["labels"].cpu()

    scores = prediction["scores"].cpu()

    masks = prediction["masks"].cpu()

    draw = ImageDraw.Draw(result)

    for i in range(len(scores)):

        score = float(scores[i])

        if score < score_threshold:

            continue

        label = int(labels[i])

        if label == 0:

            continue

        color = CLASS_COLORS[label]

        mask = (
            masks[i, 0].numpy()
            >= config.MASK_THRESHOLD
        )

        result = draw_mask(
            result,
            mask,
            color
        )

        draw = ImageDraw.Draw(result)

        box = boxes[i].numpy().tolist()

        draw.rectangle(
            box,
            outline=color,
            width=3
        )

        draw.text(
            (box[0], max(0, box[1] - 15)),

            f"{config.CLASS_NAMES[label]} {score:.2f}",

            fill=color
        )

    return result


# ============================================================
# 5. Save Comparison
# ============================================================

def save_visualization(
    image_tensor,
    target,
    prediction,
    save_path
):

    image_array = (
        image_tensor.cpu()
        .permute(1, 2, 0)
        .numpy()
        * 255
    )

    image_array = np.clip(
        image_array,
        0,
        255
    ).astype(np.uint8)

    original = Image.fromarray(
        image_array
    )

    gt_image = draw_ground_truth(
        original,
        target
    )

    pred_image = draw_predictions(
        original,
        prediction
    )

    width, height = original.size

    canvas = Image.new(
        "RGB",
        (width * 2, height)
    )

    canvas.paste(
        gt_image,
        (0, 0)
    )

    canvas.paste(
        pred_image,
        (width, 0)
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (10, 10),
        "Ground Truth",
        fill="yellow"
    )

    draw.text(
        (width + 10, 10),
        "Prediction",
        fill="yellow"
    )

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    canvas.save(save_path)
