import math

import torch
import torch.nn as nn

from torch.amp import autocast
from tqdm import tqdm

import sentencepiece as spm

import config

from model import build_model
from dataset import get_dataloaders

from utils import (
    get_device,
    load_checkpoint,
    greedy_decode
)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    amp_enabled = (
        config.USE_AMP
        and device.type == "cuda"
    )

    for batch in tqdm(
        loader,
        desc="Testing"
    ):

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

    test_loss = total_loss / max(total_tokens, 1)

    return test_loss


@torch.no_grad()
def show_translation_examples(
    model,
    dataset,
    tokenizer,
    device,
    num_examples=5
):

    model.eval()

    print("\n" + "=" * 60)
    print("Translation Examples")
    print("=" * 60)

    count = min(
        num_examples,
        len(dataset)
    )

    for i in range(count):

        sample = dataset[i]

        source = sample["source"].unsqueeze(0).to(device)

        prediction = greedy_decode(
            model,
            source
        )

        # Remove EOS from displayed IDs
        if config.EOS_ID in prediction:
            prediction = prediction[
                :prediction.index(config.EOS_ID)
            ]

        predicted_text = tokenizer.decode(
            prediction
        )

        target_ids = sample["target_output"].tolist()

        if config.EOS_ID in target_ids:
            target_ids = target_ids[
                :target_ids.index(config.EOS_ID)
            ]

        reference_text = tokenizer.decode(
            target_ids
        )

        source_text = tokenizer.decode(
            sample["source"].tolist()[:-1]
        )

        print(f"\nExample {i + 1}")
        print(f"English: {source_text}")
        print(f"Reference: {reference_text}")
        print(f"Prediction: {predicted_text}")


def main():

    device = get_device()

    print("=" * 60)
    print("Transformer Testing")
    print("=" * 60)

    print(f"Device: {device}")

    # Load datasets
    _, _, test_loader = get_dataloaders()

    # Build model
    model = build_model().to(device)

    # Load best checkpoint
    if not config.BEST_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {config.BEST_CHECKPOINT}"
        )

    checkpoint = load_checkpoint(
        config.BEST_CHECKPOINT,
        model,
        device=device
    )

    print(
        f"Loaded checkpoint from epoch: "
        f"{checkpoint['epoch']}"
    )

    # Evaluate
    criterion = nn.CrossEntropyLoss(
        ignore_index=config.PAD_ID
    )

    test_loss = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    test_perplexity = math.exp(
        min(test_loss, 20)
    )

    print("\nTest Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_perplexity:.4f}")

    # Translation examples
    tokenizer = spm.SentencePieceProcessor(
        model_file=str(config.TOKENIZER_MODEL)
    )

    show_translation_examples(
        model,
        test_loader.dataset,
        tokenizer,
        device
    )


if __name__ == "__main__":
    main()