import sentencepiece as spm
import torch

from torch.utils.data import Dataset, DataLoader

import config


class TranslationDataset(Dataset):
    """
    English-Chinese translation dataset.
    """

    def __init__(self, split):

        self.split = split

        source_file = (
            config.PROCESSED_DATA_ROOT /
            f"{split}.en"
        )

        target_file = (
            config.PROCESSED_DATA_ROOT /
            f"{split}.zh"
        )

        if not source_file.exists():
            raise FileNotFoundError(source_file)

        if not target_file.exists():
            raise FileNotFoundError(target_file)

        self.tokenizer = spm.SentencePieceProcessor(
            model_file=str(config.TOKENIZER_MODEL)
        )

        with open(source_file, "r", encoding="utf-8") as f:
            source_lines = f.read().splitlines()

        with open(target_file, "r", encoding="utf-8") as f:
            target_lines = f.read().splitlines()

        if len(source_lines) != len(target_lines):
            raise ValueError(
                f"{split} source/target size mismatch: "
                f"{len(source_lines)} != {len(target_lines)}"
            )

        self.samples = []

        self.filtered_count = 0

        for source, target in zip(source_lines, target_lines):

            source_ids = self.tokenizer.encode(
                source,
                out_type=int
            )

            target_ids = self.tokenizer.encode(
                target,
                out_type=int
            )

            # Source: tokens + EOS
            source_ids = source_ids + [
                config.EOS_ID
            ]

            # Target input: BOS + tokens
            target_input = [
                config.BOS_ID
            ] + target_ids

            # Target output: tokens + EOS
            target_output = target_ids + [
                config.EOS_ID
            ]

            if (
                len(source_ids) > config.MAX_SEQ_LEN
                or len(target_input) > config.MAX_SEQ_LEN
                or len(target_output) > config.MAX_SEQ_LEN
            ):
                self.filtered_count += 1
                continue

            self.samples.append(
                (
                    source_ids,
                    target_input,
                    target_output
                )
            )

        print(
            f"[{split}] Loaded: {len(self.samples):,}, "
            f"Filtered: {self.filtered_count:,}"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        source_ids, target_input, target_output = (
            self.samples[index]
        )

        return {
            "source": torch.tensor(
                source_ids,
                dtype=torch.long
            ),

            "target_input": torch.tensor(
                target_input,
                dtype=torch.long
            ),

            "target_output": torch.tensor(
                target_output,
                dtype=torch.long
            ),
        }


def collate_fn(batch):
    """
    Dynamically pad sequences in one batch.
    """

    batch_size = len(batch)

    max_source_len = max(
        len(item["source"])
        for item in batch
    )

    max_target_len = max(
        len(item["target_input"])
        for item in batch
    )

    source = torch.full(
        (batch_size, max_source_len),
        config.PAD_ID,
        dtype=torch.long
    )

    target_input = torch.full(
        (batch_size, max_target_len),
        config.PAD_ID,
        dtype=torch.long
    )

    target_output = torch.full(
        (batch_size, max_target_len),
        config.PAD_ID,
        dtype=torch.long
    )

    for i, item in enumerate(batch):

        source_len = len(item["source"])
        target_len = len(item["target_input"])

        source[i, :source_len] = item["source"]

        target_input[i, :target_len] = (
            item["target_input"]
        )

        target_output[i, :target_len] = (
            item["target_output"]
        )

    return {
        "source": source,
        "target_input": target_input,
        "target_output": target_output,
    }


def get_dataloaders():

    train_dataset = TranslationDataset("train")
    val_dataset = TranslationDataset("val")
    test_dataset = TranslationDataset("test")

    pin_memory = (
        config.PIN_MEMORY
        and torch.cuda.is_available()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader