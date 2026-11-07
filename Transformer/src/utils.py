import csv
import random

import numpy as np
import torch

import config


def set_seed(seed=config.RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_loss
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }

    torch.save(
        checkpoint,
        path
    )


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


@torch.no_grad()
def greedy_decode(
    model,
    source,
    max_len=config.MAX_SEQ_LEN
):
    """
    Greedy decoding for one source sentence.

    source: [1, source_len]
    """

    model.eval()

    source_mask = model.make_source_mask(source)

    encoder_output = model.encoder(
        source,
        source_mask
    )

    target = torch.tensor(
        [[config.BOS_ID]],
        dtype=torch.long,
        device=source.device
    )

    for _ in range(max_len - 1):

        decoder_output = model.decoder(
            target,
            encoder_output,
            model.make_target_mask(target),
            source_mask
        )

        logits = model.generator(
            decoder_output
        )

        # Only predict the next token
        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True
        )

        target = torch.cat(
            [target, next_token],
            dim=1
        )

        if next_token.item() == config.EOS_ID:
            break

    # Remove BOS
    return target[0, 1:].tolist()


def append_train_log(
    epoch,
    train_loss,
    val_loss,
    learning_rate
):

    config.LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = config.TRAIN_LOG.exists()

    with open(
        config.TRAIN_LOG,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "learning_rate"
            ])

        writer.writerow([
            epoch,
            train_loss,
            val_loss,
            learning_rate
        ])