from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

import config
from dataset import OxfordPetSegmentationDataset
from model import UNet
from utils import (
    calculate_binary_metrics,
    ensure_directories,
    load_checkpoint,
    set_seed,
)


def save_visualization(image_tensor, mask_tensor, probability_tensor, output_path):
    """Save a side-by-side image: input, ground truth, predicted mask."""
    image = image_tensor.cpu().permute(1, 2, 0).numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    ground_truth = (mask_tensor.cpu().squeeze(0).numpy() * 255).astype(np.uint8)
    prediction = (probability_tensor.cpu().squeeze(0).numpy() * 255).astype(np.uint8)

    image_pil = Image.fromarray(image, mode="RGB")
    mask_pil = Image.fromarray(ground_truth, mode="L").convert("RGB")
    pred_pil = Image.fromarray(prediction, mode="L").convert("RGB")

    canvas = Image.new(
        "RGB",
        (config.IMAGE_WIDTH * 3, config.IMAGE_HEIGHT),
    )
    canvas.paste(image_pil, (0, 0))
    canvas.paste(mask_pil, (config.IMAGE_WIDTH, 0))
    canvas.paste(pred_pil, (config.IMAGE_WIDTH * 2, 0))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    ensure_directories()
    set_seed()

    checkpoint_path = config.CHECKPOINT_DIR / "unet_oxford_pet_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first, or update the checkpoint path in config.py/test.py."
        )

    test_dataset = OxfordPetSegmentationDataset(split="test", augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model = UNet().to(config.DEVICE)
    checkpoint = load_checkpoint(model, checkpoint_path, config.DEVICE)
    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    total_items = 0
    saved_count = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(config.DEVICE, non_blocking=True)
            masks = masks.to(config.DEVICE, non_blocking=True)
            logits = model(images)
            probabilities = torch.sigmoid(logits)

            dice, iou = calculate_binary_metrics(logits, masks)
            batch_size = images.size(0)
            total_dice += dice * batch_size
            total_iou += iou * batch_size
            total_items += batch_size

            for index in range(batch_size):
                if saved_count >= config.SAVE_PREDICTION_LIMIT:
                    break
                save_visualization(
                    images[index],
                    masks[index],
                    (probabilities[index] >= config.THRESHOLD).float(),
                    config.PREDICTION_DIR / f"prediction_{saved_count:04d}.png",
                )
                saved_count += 1

    denominator = max(total_items, 1)
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Test samples: {total_items}")
    print(f"Test Dice: {total_dice / denominator:.4f}")
    print(f"Test IoU:  {total_iou / denominator:.4f}")
    print(f"Visualizations saved to: {config.PREDICTION_DIR}")


if __name__ == "__main__":
    main()
