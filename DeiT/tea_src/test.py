
import os

import torch
import torch.nn as nn

from tqdm import tqdm

from config import (
    DEVICE,
    BEST_CHECKPOINT,
    AMP
)

from dataset import build_dataloaders

from model import build_model


# ============================================================
# 1. Test
# ============================================================

@torch.no_grad()
def test(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(
        loader,
        desc="Test"
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=AMP and device.type == "cuda"
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ============================================================
# 2. Main
# ============================================================

def main():

    print("=" * 60)
    print("ResNet-18 Teacher Test")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    _, _, test_loader = build_dataloaders()

    print(f"Test samples: {len(test_loader.dataset)}")

    class_to_idx = test_loader.dataset.class_to_idx

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model().to(DEVICE)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    if not os.path.exists(BEST_CHECKPOINT):

        raise FileNotFoundError(
            f"Checkpoint not found: {BEST_CHECKPOINT}"
        )

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if checkpoint["class_to_idx"] != class_to_idx:

        raise ValueError(
            "Checkpoint class mapping does not match test dataset!"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("\nLoaded checkpoint:")

    print(f"Epoch: {checkpoint['epoch'] + 1}")

    print(
        f"Best validation loss: "
        f"{checkpoint['best_val_loss']:.6f}"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = test(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print()

    print(f"Test Loss: {test_loss:.4f}")

    print(f"Test Accuracy: {test_acc:.4f}")

    print()
    print("=" * 60)
    print("Test Finished")
    print("=" * 60)


if __name__ == "__main__":
    main()
