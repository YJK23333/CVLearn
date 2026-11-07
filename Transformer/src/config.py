from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CVLEARN_ROOT = PROJECT_ROOT.parent

DATA_ROOT = CVLEARN_ROOT / "data" / "ted2020"

RAW_DATA_ROOT = DATA_ROOT / "raw" / "en-zh_cn"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed" / "en-zh_cn"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"


# ============================================================
# Raw dataset files
# ============================================================

SOURCE_FILE = RAW_DATA_ROOT / "TED2020.en-zh_cn.en"
TARGET_FILE = RAW_DATA_ROOT / "TED2020.en-zh_cn.zh_cn"


# ============================================================
# Dataset split
# ============================================================

TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

RANDOM_SEED = 42


# ============================================================
# Tokenizer
# ============================================================

TOKENIZER_DIR = PROCESSED_DATA_ROOT / "tokenizer"

TOKENIZER_MODEL = TOKENIZER_DIR / "sentencepiece.model"

VOCAB_SIZE = 8000
MODEL_TYPE = "bpe"
CHARACTER_COVERAGE = 0.9995

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


# ============================================================
# Dataset
# ============================================================

MAX_SEQ_LEN = 64

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

NUM_WORKERS = 0
PIN_MEMORY = True


# ============================================================
# Transformer model
# ============================================================

D_MODEL = 256

NUM_HEADS = 4

NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

D_FF = 1024

DROPOUT = 0.1

MAX_POSITIONAL_ENCODING = MAX_SEQ_LEN


# ============================================================
# Training
# ============================================================

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

GRAD_CLIP = 1.0

USE_AMP = True

# 每隔多少个 epoch 输出一次训练信息
LOG_INTERVAL = 100

# 是否从最近一次 checkpoint 恢复
RESUME = True


# ============================================================
# Checkpoints
# ============================================================

LAST_CHECKPOINT = CHECKPOINT_DIR / "last.pt"

BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"


# ============================================================
# Logs
# ============================================================

TRAIN_LOG = LOG_DIR / "train.csv"