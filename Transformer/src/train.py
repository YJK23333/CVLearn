
import math
import time

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

from tqdm import tqdm

import config

from model import build_model
from dataset import get_dataloaders

from utils import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    append_train_log,
)


# ============================================================
# Training one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    epoch
):
    """
    Train the model for one epoch.

    Returns:
        epoch_loss: token-level average cross entropy loss
    """

    model.train()

    total_loss = 0.0
    total_tokens = 0

    amp_enabled = (
        config.USE_AMP
        and device.type == "cuda"
    )

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch} Training",
        leave=False
    )

    for batch_idx, batch in enumerate(progress):

        # ----------------------------------------------------
        # 1. Move data to device
        # ----------------------------------------------------

        source = batch["source"].to(device)

        target_input = batch["target_input"].to(device)

        target_output = batch["target_output"].to(device)

        # ----------------------------------------------------
        # 2. Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------------------
        # 3. Forward propagation
        # ----------------------------------------------------

        with autocast(
            device_type=device.type,
            enabled=amp_enabled
        ):

            logits = model(
                source,
                target_input
            )

            loss = criterion(
                logits.reshape(-1, config.VOCAB_SIZE),
                target_output.reshape(-1)
            )

        # ----------------------------------------------------
        # 4. Backward propagation
        # ----------------------------------------------------

        scaler.scale(loss).backward()

        # Unscale gradients before gradient clipping
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.GRAD_CLIP
        )

        # ----------------------------------------------------
        # 5. Optimizer update
        # ----------------------------------------------------

        scaler.step(optimizer)

        scaler.update()

        # ----------------------------------------------------
        # 6. Accumulate token-level loss
        # ----------------------------------------------------

        valid_tokens = (
            target_output != config.PAD_ID
        ).sum().item()

        total_loss += loss.detach().item() * valid_tokens

        total_tokens += valid_tokens

        # ----------------------------------------------------
        # 7. Update progress bar
        # ----------------------------------------------------

        if (batch_idx + 1) % config.LOG_INTERVAL == 0:

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": (
                    f"{optimizer.param_groups[0]['lr']:.2e}"
                )
            })

    epoch_loss = total_loss / max(total_tokens, 1)

    return epoch_loss


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    epoch
):
    """
    Evaluate the model on validation dataset.

    No gradient computation is performed.
    """

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    amp_enabled = (
        config.USE_AMP
        and device.type == "cuda"
    )

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch} Validation",
        leave=False
    )

    for batch in progress:

        source = batch["source"].to(device)

        target_input = batch["target_input"].to(device)

        target_output = batch["target_output"].to(device)

        with autocast(
            device_type=device.type,
            enabled=amp_enabled
        ):

            logits = model(
                source,
                target_input
            )

            loss = criterion(
                logits.reshape(-1, config.VOCAB_SIZE),
                target_output.reshape(-1)
            )

        valid_tokens = (
            target_output != config.PAD_ID
        ).sum().item()

        total_loss += loss.item() * valid_tokens

        total_tokens += valid_tokens

    val_loss = total_loss / max(total_tokens, 1)

    return val_loss


# ============================================================
# Model information
# ============================================================

def print_model_info(model):

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


# ============================================================
# Checkpoint recovery
# ============================================================

def resume_training(
    model,
    optimizer,
    scheduler,
    scaler,
    device
):
    """
    Restore training states from latest checkpoint.

    Returns:
        start_epoch
        best_val_loss
    """

    if not config.RESUME:

        return 1, float("inf")

    if not config.LAST_CHECKPOINT.exists():

        print(
            "No checkpoint found. "
            "Starting training from scratch."
        )

        return 1, float("inf")

    print("\nResuming training...")

    checkpoint = load_checkpoint(
        config.LAST_CHECKPOINT,
        model,
        optimizer,
        scheduler,
        scaler,
        device
    )

    start_epoch = checkpoint["epoch"] + 1

    best_val_loss = checkpoint["best_val_loss"]

    print(f"Resumed epoch: {checkpoint['epoch']}")

    print(f"Best validation loss: {best_val_loss:.4f}")

    return start_epoch, best_val_loss


# ============================================================
# Save checkpoints
# ============================================================

def save_training_checkpoints(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    val_loss,
    best_val_loss
):
    """
    Save best and latest checkpoints.

    The best checkpoint is updated only when
    validation loss improves.
    """

    is_best = val_loss < best_val_loss

    if is_best:

        best_val_loss = val_loss

        save_checkpoint(
            config.BEST_CHECKPOINT,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_loss
        )

        print("Best checkpoint updated.")

    # Always save the latest training state
    save_checkpoint(
        config.LAST_CHECKPOINT,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        best_val_loss
    )

    return best_val_loss


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Initialize
    # --------------------------------------------------------

    set_seed()

    config.CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    config.LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = get_device()

    print("=" * 65)
    print("Transformer English-Chinese Translation Training")
    print("=" * 65)

    print(f"Device: {device}")

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(device)}"
        )

        print(
            f"GPU Memory: "
            f"{torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.2f} GB"
        )

    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.TRAIN_BATCH_SIZE}")
    print(f"Maximum sequence length: {config.MAX_SEQ_LEN}")
    print(f"Mixed precision: {config.USE_AMP}")

    # --------------------------------------------------------
    # 2. Load datasets
    # --------------------------------------------------------

    print("\n[1/4] Loading datasets...")

    train_loader, val_loader, _ = get_dataloaders()

    if len(train_loader) == 0:

        raise RuntimeError(
            "Training DataLoader is empty."
        )

    if len(val_loader) == 0:

        raise RuntimeError(
            "Validation DataLoader is empty."
        )

    print(f"Training batches: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")

    # --------------------------------------------------------
    # 3. Build model
    # --------------------------------------------------------

    print("\n[2/4] Building Transformer...")

    model = build_model().to(device)

    print_model_info(model)

    # --------------------------------------------------------
    # 4. Initialize optimizer and loss
    # --------------------------------------------------------

    print("\n[3/4] Initializing training components...")

    optimizer = Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=config.PAD_ID
    )

    amp_enabled = (
        config.USE_AMP
        and device.type == "cuda"
    )

    scaler = GradScaler(
        "cuda",
        enabled=amp_enabled
    )

    # --------------------------------------------------------
    # 5. Resume training if enabled
    # --------------------------------------------------------

    start_epoch, best_val_loss = resume_training(
        model,
        optimizer,
        scheduler,
        scaler,
        device
    )

    if start_epoch > config.EPOCHS:

        print(
            "Training has already reached the configured "
            "number of epochs."
        )

        return

    # --------------------------------------------------------
    # 6. Training loop
    # --------------------------------------------------------

    print("\n[4/4] Starting training...")

    for epoch in range(
        start_epoch,
        config.EPOCHS + 1
    ):

        epoch_start_time = time.time()

        print("\n" + "=" * 65)

        print(
            f"Epoch [{epoch}/{config.EPOCHS}]"
        )

        print("=" * 65)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            epoch
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch
        )

        # ----------------------------------------------------
        # Learning rate scheduling
        # ----------------------------------------------------

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Calculate perplexity
        # ----------------------------------------------------

        train_perplexity = math.exp(
            min(train_loss, 20)
        )

        val_perplexity = math.exp(
            min(val_loss, 20)
        )

        epoch_time = time.time() - epoch_start_time

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\nEpoch Results")

        print(
            f"Train Loss:       {train_loss:.4f}"
        )

        print(
            f"Validation Loss:  {val_loss:.4f}"
        )

        print(
            f"Train Perplexity: {train_perplexity:.4f}"
        )

        print(
            f"Val Perplexity:   {val_perplexity:.4f}"
        )

        print(
            f"Learning Rate:    {current_lr:.8f}"
        )

        print(
            f"Epoch Time:       {epoch_time / 60:.2f} min"
        )

        # ----------------------------------------------------
        # Save CSV log
        # ----------------------------------------------------

        append_train_log(
            epoch,
            train_loss,
            val_loss,
            current_lr
        )

        # ----------------------------------------------------
        # Save checkpoints
        # ----------------------------------------------------

        best_val_loss = save_training_checkpoints(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            val_loss,
            best_val_loss
        )

        print(
            f"Best Validation Loss: {best_val_loss:.4f}"
        )

    # --------------------------------------------------------
    # Training finished
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("Training Completed")
    print("=" * 65)

    print(
        f"Best Validation Loss: {best_val_loss:.4f}"
    )

    print(
        f"Best Checkpoint: {config.BEST_CHECKPOINT}"
    )

    print(
        f"Latest Checkpoint: {config.LAST_CHECKPOINT}"
    )


if __name__ == "__main__":

    main()
