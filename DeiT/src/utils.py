
import os
import random

import numpy as np
import torch


# ============================================================
# 1. Random Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Accuracy
# ============================================================

def calculate_accuracy(outputs, targets):

    predictions = outputs.argmax(dim=1)

    correct = (
        predictions == targets
    ).sum().item()

    return correct / targets.size(0)


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
    class_to_idx
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "best_val_loss": best_val_loss,

        "class_to_idx": class_to_idx
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# 4. Load Checkpoint
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device="cpu"
):

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None:

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if scaler is not None:

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    return checkpoint
