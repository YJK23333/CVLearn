
import os
import random

import numpy as np
import torch

from tqdm import tqdm

import config


# ============================================================
# 1. Device
# ============================================================

def get_device():

    device = torch.device(config.DEVICE)

    print(f"Device: {device}")

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    return device


# ============================================================
# 2. Move Targets to Device
# ============================================================

def move_targets_to_device(targets, device):

    return [

        {
            key: value.to(device)
            for key, value in target.items()
        }

        for target in targets
    ]


# ============================================================
# 3. Save Checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_loss,
    loader_generator
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "best_val_loss": best_val_loss,

        "python_rng_state": random.getstate(),

        "numpy_rng_state": np.random.get_state(),

        "torch_rng_state": torch.get_rng_state(),

        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),

        "loader_generator_state": (
            loader_generator.get_state()
        ),

        "config": {

            "epochs": config.EPOCHS,

            "num_classes": config.NUM_CLASSES,

            "image_min_size": config.IMAGE_MIN_SIZE,

            "image_max_size": config.IMAGE_MAX_SIZE,

            "seed": config.SEED
        }
    }

    temporary_path = path + ".tmp"

    torch.save(
        checkpoint,
        temporary_path
    )

    os.replace(
        temporary_path,
        path
    )


# ============================================================
# 4. Load Checkpoint
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    loader_generator
):

    # --------------------------------------------------------
    # Load Checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    # --------------------------------------------------------
    # Restore Model
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Restore Optimizer
    # --------------------------------------------------------

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # --------------------------------------------------------
    # Restore Scheduler
    # --------------------------------------------------------

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    # --------------------------------------------------------
    # Restore AMP Scaler
    # --------------------------------------------------------

    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

    # --------------------------------------------------------
    # Restore Random States
    # --------------------------------------------------------

    random.setstate(
        checkpoint["python_rng_state"]
    )

    np.random.set_state(
        checkpoint["numpy_rng_state"]
    )

    torch.set_rng_state(
        checkpoint["torch_rng_state"].cpu()
    )

    cuda_rng_state = checkpoint.get(
        "cuda_rng_state",
        None
    )

    if (
        torch.cuda.is_available()
        and cuda_rng_state is not None
    ):

        # Ensure every CUDA RNG state is a CPU ByteTensor.

        cuda_rng_state = [

            state.cpu().to(torch.uint8)

            for state in cuda_rng_state
        ]

        torch.cuda.set_rng_state_all(
            cuda_rng_state
        )

    # --------------------------------------------------------
    # Restore DataLoader Generator
    # --------------------------------------------------------

    loader_generator.set_state(
        checkpoint["loader_generator_state"].cpu()
    )

    # --------------------------------------------------------
    # Resume Information
    # --------------------------------------------------------

    start_epoch = checkpoint["epoch"] + 1

    best_val_loss = checkpoint["best_val_loss"]

    print("=" * 60)

    print("Checkpoint loaded successfully.")

    print(f"Last completed epoch: {checkpoint['epoch']}")

    print(f"Resume from epoch: {start_epoch}")

    print(f"Best validation loss: {best_val_loss:.6f}")

    print("=" * 60)

    return start_epoch, best_val_loss


# ============================================================
# 5. Loss Meter
# ============================================================

class AverageMeter:

    def __init__(self):

        self.reset()

    def reset(self):

        self.total = 0.0

        self.count = 0

    def update(self, value, n=1):

        self.total += value * n

        self.count += n

    @property
    def average(self):

        if self.count == 0:

            return 0.0

        return self.total / self.count


# ============================================================
# 6. Print GPU Memory
# ============================================================

def print_gpu_memory():

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated(0) / 1024 ** 3
        )

        reserved = (
            torch.cuda.memory_reserved(0) / 1024 ** 3
        )

        print(
            f"GPU Memory: "
            f"{allocated:.2f} GB allocated / "
            f"{reserved:.2f} GB reserved"
        )
