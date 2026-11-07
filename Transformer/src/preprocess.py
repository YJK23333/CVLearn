import random
import json
import unicodedata

from pathlib import Path

import config


def read_parallel_corpus(source_file, target_file):
    """
    Read source and target files.

    Each line in the source file corresponds to the
    same line in the target file.
    """

    with open(source_file, "r", encoding="utf-8") as f:
        source_lines = f.readlines()

    with open(target_file, "r", encoding="utf-8") as f:
        target_lines = f.readlines()

    if len(source_lines) != len(target_lines):
        raise ValueError(
            f"Line count mismatch: "
            f"{len(source_lines)} != {len(target_lines)}"
        )

    return source_lines, target_lines


def normalize_text(text):
    """
    Basic text normalization.
    """

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Remove leading/trailing whitespace
    # and normalize consecutive whitespace
    text = " ".join(text.split())

    return text


def clean_parallel_corpus(source_lines, target_lines):
    """
    Clean bilingual sentence pairs.

    Operations:
    1. Normalize whitespace and Unicode.
    2. Remove empty sentence pairs.
    3. Remove exact duplicate sentence pairs.
    """

    cleaned_pairs = []
    seen_pairs = set()

    empty_count = 0
    duplicate_count = 0

    for source, target in zip(source_lines, target_lines):

        source = normalize_text(source)
        target = normalize_text(target)

        # Remove empty sentence pairs
        if not source or not target:
            empty_count += 1
            continue

        pair = (source, target)

        # Remove exact duplicate pairs
        if pair in seen_pairs:
            duplicate_count += 1
            continue

        seen_pairs.add(pair)
        cleaned_pairs.append(pair)

    stats = {
        "original_pairs": len(source_lines),
        "empty_pairs": empty_count,
        "duplicate_pairs": duplicate_count,
        "cleaned_pairs": len(cleaned_pairs),
    }

    return cleaned_pairs, stats


def split_dataset(pairs):
    """
    Split dataset into train, validation and test sets.
    """

    pairs = pairs.copy()

    rng = random.Random(config.RANDOM_SEED)
    rng.shuffle(pairs)

    total = len(pairs)

    train_size = int(total * config.TRAIN_RATIO)
    val_size = int(total * config.VAL_RATIO)

    train_pairs = pairs[:train_size]

    val_pairs = pairs[
        train_size:train_size + val_size
    ]

    test_pairs = pairs[
        train_size + val_size:
    ]

    return train_pairs, val_pairs, test_pairs


def save_parallel_corpus(pairs, source_path, target_path):
    """
    Save bilingual sentence pairs into two aligned files.
    """

    source_path.parent.mkdir(parents=True, exist_ok=True)

    with open(source_path, "w", encoding="utf-8", newline="\n") as src_file, \
         open(target_path, "w", encoding="utf-8", newline="\n") as tgt_file:

        for source, target in pairs:
            src_file.write(source + "\n")
            tgt_file.write(target + "\n")


def main():

    print("=" * 60)
    print("TED2020 English-Chinese Preprocessing")
    print("=" * 60)

    # 1. Read raw corpus
    print("\n[1/4] Reading raw corpus...")

    source_lines, target_lines = read_parallel_corpus(
        config.SOURCE_FILE,
        config.TARGET_FILE
    )

    print(f"Original sentence pairs: {len(source_lines):,}")

    # 2. Clean corpus
    print("\n[2/4] Cleaning corpus...")

    cleaned_pairs, stats = clean_parallel_corpus(
        source_lines,
        target_lines
    )

    for key, value in stats.items():
        print(f"{key}: {value:,}")

    # 3. Split dataset
    print("\n[3/4] Splitting dataset...")

    train_pairs, val_pairs, test_pairs = split_dataset(
        cleaned_pairs
    )

    print(f"Train:      {len(train_pairs):,}")
    print(f"Validation: {len(val_pairs):,}")
    print(f"Test:       {len(test_pairs):,}")

    # 4. Save processed corpus
    print("\n[4/4] Saving processed corpus...")

    output_dir = config.PROCESSED_DATA_ROOT

    save_parallel_corpus(
        train_pairs,
        output_dir / "train.en",
        output_dir / "train.zh"
    )

    save_parallel_corpus(
        val_pairs,
        output_dir / "val.en",
        output_dir / "val.zh"
    )

    save_parallel_corpus(
        test_pairs,
        output_dir / "test.en",
        output_dir / "test.zh"
    )

    # Save preprocessing statistics
    stats.update({
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "random_seed": config.RANDOM_SEED,
    })

    with open(
        output_dir / "stats.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    print("\nPreprocessing completed.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()