
import torch
import numpy as np

from PIL import Image

from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

import config
from model import build_model
from dataset import get_dataloaders


@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()

    total_intersection = 0.0
    total_pred = 0.0
    total_target = 0.0
    total_correct = 0
    total_pixels = 0

    for images, masks in tqdm(loader, desc="Testing"):

        images = images.to(device)
        masks = masks.to(device)

        with autocast(
            device_type="cuda",
            enabled=config.USE_AMP and device.type == "cuda"
        ):

            outputs = model(images)

        preds = (
            torch.sigmoid(outputs) >= config.THRESHOLD
        ).long()

        targets = masks.long()

        total_intersection += (
            (preds * targets).sum().item()
        )

        total_pred += preds.sum().item()
        total_target += targets.sum().item()

        total_correct += (
            (preds == targets).sum().item()
        )

        total_pixels += targets.numel()

    dice = (
        2 * total_intersection + 1e-6
    ) / (
        total_pred + total_target + 1e-6
    )

    union = total_pred + total_target - total_intersection

    iou = (
        total_intersection + 1e-6
    ) / (
        union + 1e-6
    )

    pixel_accuracy = total_correct / total_pixels

    return {
        "dice": dice,
        "iou": iou,
        "pixel_accuracy": pixel_accuracy
    }


@torch.no_grad()
def save_predictions(model, loader, device):

    model.eval()

    output_dir = config.PREDICTION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_index = 0

    for images, masks in tqdm(loader, desc="Saving predictions"):

        images = images.to(device)

        with autocast(
            device_type="cuda",
            enabled=config.USE_AMP and device.type == "cuda"
        ):

            outputs = model(images)

        preds = (
            torch.sigmoid(outputs) >= config.THRESHOLD
        ).float()

        images = images.cpu()
        masks = masks.cpu()
        preds = preds.cpu()

        for i in range(images.shape[0]):

            image = (
                images[i].permute(1, 2, 0).numpy() * 255
            ).clip(0, 255).astype(np.uint8)

            mask = (
                masks[i, 0].numpy() * 255
            ).astype(np.uint8)

            pred = (
                preds[i, 0].numpy() * 255
            ).astype(np.uint8)

            image_pil = Image.fromarray(image)
            mask_pil = Image.fromarray(mask)
            pred_pil = Image.fromarray(pred)

            # Save individual results
            image_pil.save(
                output_dir / f"{sample_index}_image.png"
            )

            mask_pil.save(
                output_dir / f"{sample_index}_gt.png"
            )

            pred_pil.save(
                output_dir / f"{sample_index}_pred.png"
            )

            # Create comparison image:
            # Original | Ground Truth | Prediction
            comparison = Image.new(
                "RGB",
                (image_pil.width * 3, image_pil.height)
            )

            comparison.paste(image_pil, (0, 0))
            comparison.paste(
                mask_pil.convert("RGB"),
                (image_pil.width, 0)
            )
            comparison.paste(
                pred_pil.convert("RGB"),
                (image_pil.width * 2, 0)
            )

            comparison.save(
                output_dir / f"{sample_index}_comparison.png"
            )

            sample_index += 1


def main():

    device = config.DEVICE

    print("=" * 50)
    print("U-Net Baseline Testing")
    print(f"Device: {device}")
    print("=" * 50)

    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: "
            f"{config.BEST_MODEL_PATH}"
        )

    # Dataset
    _, _, test_loader = get_dataloaders()

    # Model
    model = build_model().to(device)

    # Load best checkpoint
    checkpoint = torch.load(
        config.BEST_MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(f"Loaded checkpoint: {config.BEST_MODEL_PATH}")
    print(f"Best epoch: {checkpoint['epoch']}")

    # Evaluation
    metrics = evaluate(
        model,
        test_loader,
        device
    )

    print("\n" + "=" * 50)
    print("Final Test Results")
    print("=" * 50)

    print(f"Dice:            {metrics['dice']:.4f}")
    print(f"IoU:             {metrics['iou']:.4f}")
    print(
        f"Pixel Accuracy:  "
        f"{metrics['pixel_accuracy']:.4f}"
    )

    print("=" * 50)

    # Save metrics
    result_path = config.RUN_DIR / "test_results.txt"

    with open(result_path, "w", encoding="utf-8") as f:

        f.write("U-Net Baseline Test Results\n")
        f.write(f"Dice: {metrics['dice']:.6f}\n")
        f.write(f"IoU: {metrics['iou']:.6f}\n")
        f.write(
            f"Pixel Accuracy: "
            f"{metrics['pixel_accuracy']:.6f}\n"
        )

    # Save visualizations
    if config.SAVE_PREDICTIONS:

        save_predictions(
            model,
            test_loader,
            device
        )

    print(f"\nResults saved to: {config.RUN_DIR}")


if __name__ == "__main__":
    main()
