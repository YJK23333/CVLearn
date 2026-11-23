
import torch

from torch.amp import autocast
from tqdm import tqdm

import config

from dataset import get_dataloaders
from model import FasterRCNN

from utils import (
    load_checkpoint,
    postprocess_detections,
    evaluate_voc,
)


@torch.no_grad()
def test():

    device = torch.device(config.DEVICE)

    _, _, test_loader = get_dataloaders()

    model = FasterRCNN(
        pretrained=False
    ).to(device)

    checkpoint = torch.load(
        config.BEST_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    print("Loaded checkpoint:")
    print("Epoch:", checkpoint["epoch"] + 1)
    print("Best validation loss:", checkpoint["best_loss"])

    predictions = {}
    ground_truths = {}

    for images, targets in tqdm(
        test_loader,
        desc="Testing"
    ):

        images = [
            image.to(device)
            for image in images
        ]

        with autocast(
            device_type=device.type,
            enabled=config.AMP and device.type == "cuda"
        ):

            outputs = model(images)

        image_size = images[0].shape[-2:]

        result = postprocess_detections(
            outputs,
            image_size
        )

        image_id = targets[0]["image_id"]

        predictions[image_id] = {
            "boxes": result["boxes"].cpu(),
            "scores": result["scores"].cpu(),
            "labels": result["labels"].cpu(),
        }

        ground_truths[image_id] = {
            "boxes": targets[0]["boxes"].cpu(),
            "labels": targets[0]["labels"].cpu(),
        }

    ap_per_class, mean_ap = evaluate_voc(
        predictions,
        ground_truths
    )

    print("\nAP per class:\n")

    for name, ap in ap_per_class.items():

        print(
            f"{name:<15}: {ap:.4f}"
        )

    print("\n" + "=" * 40)

    print(
        f"mAP@0.5: {mean_ap:.4f}"
    )

    print("=" * 40)

    torch.save(
        {
            "predictions": predictions,
            "ground_truths": ground_truths,
            "ap_per_class": ap_per_class,
            "mAP": mean_ap,
        },
        config.OUTPUT_DIR / "test_results.pth"
    )


if __name__ == "__main__":

    test()
