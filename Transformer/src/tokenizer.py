import sentencepiece as spm

import config


def train_tokenizer():

    print("=" * 60)
    print("Training SentencePiece Tokenizer")
    print("=" * 60)

    config.TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Training corpus: English + Chinese
    input_files = [
        config.PROCESSED_DATA_ROOT / "train.en",
        config.PROCESSED_DATA_ROOT / "train.zh",
    ]

    input_files = [
        str(path) for path in input_files
    ]

    model_prefix = str(
        config.TOKENIZER_DIR / "sentencepiece"
    )

    print("\nTraining configuration:")
    print(f"Vocabulary size: {config.VOCAB_SIZE}")
    print(f"Model type: {config.MODEL_TYPE}")
    print(f"Input files: {input_files}")

    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=config.VOCAB_SIZE,
        model_type=config.MODEL_TYPE,
        character_coverage=config.CHARACTER_COVERAGE,

        pad_id=config.PAD_ID,
        unk_id=config.UNK_ID,
        bos_id=config.BOS_ID,
        eos_id=config.EOS_ID,

        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",

        shuffle_input_sentence=True,
        input_sentence_size=0,

        normalization_rule_name="nmt_nfkc",

        num_threads=4,
    )

    print("\nTokenizer training completed.")

    print(f"Model: {model_prefix}.model")
    print(f"Vocabulary: {model_prefix}.vocab")


def test_tokenizer():

    model_path = (
        config.TOKENIZER_DIR /
        "sentencepiece.model"
    )

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(model_path)
    )

    test_sentences = [
        "I love machine learning.",
        "Transformer is an important model.",
        "我正在学习深度学习。",
        "注意力机制是 Transformer 的核心。",
    ]

    print("\n" + "=" * 60)
    print("Tokenizer Test")
    print("=" * 60)

    print(f"Vocabulary size: {tokenizer.get_piece_size()}")

    for sentence in test_sentences:

        pieces = tokenizer.encode(
            sentence,
            out_type=str
        )

        ids = tokenizer.encode(
            sentence,
            out_type=int
        )

        print(f"\nOriginal: {sentence}")
        print(f"Pieces:   {pieces}")
        print(f"Token IDs: {ids}")

        decoded = tokenizer.decode(ids)

        print(f"Decoded:  {decoded}")


if __name__ == "__main__":

    train_tokenizer()

    test_tokenizer()