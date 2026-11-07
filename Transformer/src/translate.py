import argparse

import torch
import sentencepiece as spm

import config

from model import build_model

from utils import (
    get_device,
    load_checkpoint,
    greedy_decode
)


def translate(
    text,
    model,
    tokenizer,
    device
):

    model.eval()

    source_ids = tokenizer.encode(
        text,
        out_type=int
    )

    source_ids.append(
        config.EOS_ID
    )

    if len(source_ids) > config.MAX_SEQ_LEN:

        raise ValueError(
            f"Input sequence is too long: "
            f"{len(source_ids)} > {config.MAX_SEQ_LEN}"
        )

    source = torch.tensor(
        [source_ids],
        dtype=torch.long,
        device=device
    )

    prediction = greedy_decode(
        model,
        source
    )

    if config.EOS_ID in prediction:

        prediction = prediction[
            :prediction.index(config.EOS_ID)
        ]

    return tokenizer.decode(
        prediction
    )


def main():

    parser = argparse.ArgumentParser(
        description="English to Chinese Translation"
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="English sentence to translate"
    )

    args = parser.parse_args()

    device = get_device()

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor(
        model_file=str(config.TOKENIZER_MODEL)
    )

    # Build model
    model = build_model().to(device)

    # Load checkpoint
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
        f"Loaded model from epoch: {checkpoint['epoch']}"
    )

    # Translate
    result = translate(
        args.text,
        model,
        tokenizer,
        device
    )

    print("\n" + "=" * 60)
    print("Translation")
    print("=" * 60)

    print(f"English: {args.text}")
    print(f"Chinese: {result}")


if __name__ == "__main__":
    main()