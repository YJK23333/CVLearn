
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
from torch.amp import autocast

import config

from dataset import VOCDataset
from model import FasterRCNN
from utils import postprocess_detections


# ============================================================
# Configuration
# ============================================================

NUM_IMAGES = 20

SCORE_THRESHOLD = 0.5

RANDOM_SEED = config.SEED


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = config.OUTPUT_DIR / "visualize"

CHECKPOINT_PATH = config.BEST_CHECKPOINT


# ============================================================
# Font
# ============================================================

def get_font(size=18):

    font_candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
    ]

    for font_path in font_candidates:

        if Path(font_path).exists():

            try:
                return ImageFont.truetype(
                    font_path,
                    size=size
                )

            except OSError:
                pass

    return ImageFont.load_default()


# ============================================================
# Class Name
# ============================================================

def get_class_name(label):

    label = int(label)

    if label == config.BACKGROUND_IDX:
        return "background"

    class_index = label - 1

    if not 0 <= class_index < len(config.VOC_CLASSES):

        raise ValueError(
            f"Invalid class label: {label}"
        )

    return config.VOC_CLASSES[class_index]


# ============================================================
# Drawing
# ============================================================

def draw_box(
    draw,
    box,
    text,
    color,
    font,
    width=3
):

    x1, y1, x2, y2 = [
        int(v)
        for v in box
    ]

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=color,
        width=width
    )

    if not text:
        return

    bbox = draw.textbbox(
        (x1, y1),
        text,
        font=font
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = x1

    text_y = max(
        0,
        y1 - text_height - 6
    )

    draw.rectangle(
        [
            text_x,
            text_y,
            text_x + text_width + 6,
            text_y + text_height + 6
        ],
        fill=color
    )

    draw.text(
        (text_x + 3, text_y + 3),
        text,
        fill="white",
        font=font
    )


# ============================================================
# Draw Ground Truth
# ============================================================

def draw_ground_truth(
    image,
    boxes,
    labels
):

    draw = ImageDraw.Draw(image)

    font = get_font(18)

    for box, label in zip(boxes, labels):

        class_name = get_class_name(label)

        draw_box(
            draw,
            box,
            f"GT: {class_name}",
            "green",
            font
        )


# ============================================================
# Draw Predictions
# ============================================================

def draw_predictions(
    image,
    boxes,
    scores,
    labels
):

    draw = ImageDraw.Draw(image)

    font = get_font(18)

    for box, score, label in zip(
        boxes,
        scores,
        labels
    ):

        score = float(score)

        if score < SCORE_THRESHOLD:
            continue

        class_name = get_class_name(label)

        text = (
            f"{class_name}: "
            f"{score:.2f}"
        )

        draw_box(
            draw,
            box,
            text,
            "red",
            font
        )


# ============================================================
# Image Denormalization
# ============================================================

def denormalize_image(image_tensor):

    mean = torch.tensor(
        config.IMAGE_MEAN,
        dtype=image_tensor.dtype
    ).view(3, 1, 1)

    std = torch.tensor(
        config.IMAGE_STD,
        dtype=image_tensor.dtype
    ).view(3, 1, 1)

    image = (
        image_tensor.cpu() * std + mean
    )

    image = torch.clamp(
        image,
        0,
        1
    )

    return to_pil_image(image)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(config.DEVICE)

    print(f"Device: {device}")

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = VOCDataset(
        root=config.DATA_ROOT,
        split="test",
        augment=False
    )

    print(
        f"Test dataset size: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Random Sampling
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    num_images = min(
        NUM_IMAGES,
        len(dataset)
    )

    indices = random.sample(
        range(len(dataset)),
        num_images
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FasterRCNN(
        pretrained=False
    ).to(device)

    # --------------------------------------------------------
    # Load Checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    print("\nLoaded checkpoint:")

    if "epoch" in checkpoint:

        print(
            f"Epoch: {checkpoint['epoch'] + 1}"
        )

    if "best_loss" in checkpoint:

        print(
            f"Best validation loss: "
            f"{checkpoint['best_loss']:.4f}"
        )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    print("\nStart visualization...\n")

    with torch.no_grad():

        for count, index in enumerate(
            indices,
            start=1
        ):

            # --------------------------------------------
            # Load Image
            # --------------------------------------------

            image_tensor, target = dataset[index]

            input_tensor = (
                image_tensor
                .unsqueeze(0)
                .to(device)
            )

            # --------------------------------------------
            # Forward
            # --------------------------------------------

            with autocast(
                device_type=device.type,
                enabled=(
                    config.AMP
                    and device.type == "cuda"
                )
            ):

                outputs = model(input_tensor)

            # --------------------------------------------
            # Post-process
            # --------------------------------------------

            image_size = image_tensor.shape[-2:]

            detections = postprocess_detections(
                outputs,
                image_size
            )

            pred_boxes = detections["boxes"]
            pred_scores = detections["scores"]
            pred_labels = detections["labels"]

            # --------------------------------------------
            # Recover Original Image
            # --------------------------------------------

            pil_image = denormalize_image(
                image_tensor
            )

            # --------------------------------------------
            # Ground Truth
            # --------------------------------------------

            gt_boxes = target["boxes"].cpu()
            gt_labels = target["labels"].cpu()

            draw_ground_truth(
                pil_image,
                gt_boxes,
                gt_labels
            )

            # --------------------------------------------
            # Predictions
            # --------------------------------------------

            draw_predictions(
                pil_image,
                pred_boxes.cpu(),
                pred_scores.cpu(),
                pred_labels.cpu()
            )

            # --------------------------------------------
            # Save
            # --------------------------------------------

            image_id = target["image_id"]

            output_path = (
                OUTPUT_DIR
                / f"{count:02d}_{image_id}.jpg"
            )

            pil_image.save(
                output_path,
                quality=95
            )

            # --------------------------------------------
            # Statistics
            # --------------------------------------------

            num_predictions = int(
                (
                    pred_scores >= SCORE_THRESHOLD
                ).sum()
            )

            num_ground_truth = len(gt_boxes)

            print(
                f"[{count:02d}/{num_images}] "
                f"{image_id} | "
                f"GT: {num_ground_truth} | "
                f"Pred: {num_predictions} | "
                f"Saved: {output_path}"
            )

    print("\nVisualization completed.")

    print(
        f"Results saved to:\n{OUTPUT_DIR}"
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()
