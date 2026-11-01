
from pathlib import Path
import torch


# ============================================================
# 1. Project Paths
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
ROOT_DIR = PROJECT_DIR.parent

DATASET_ROOT = ROOT_DIR / "data" / "oxford-iiit-pet"

IMAGE_DIR = DATASET_ROOT / "images"
ANNOTATION_DIR = DATASET_ROOT / "annotations"
TRIMAP_DIR = ANNOTATION_DIR / "trimaps"

TRAINVAL_FILE = ANNOTATION_DIR / "trainval.txt"
TEST_FILE = ANNOTATION_DIR / "test.txt"

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RUN_DIR = PROJECT_DIR / "runs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Dataset Configuration
# ============================================================

IMAGE_SIZE = 256

NUM_CLASSES = 1

VAL_RATIO = 0.15
RANDOM_SEED = 42

# Oxford-IIIT Pet trimap:
# 1 = Pet
# 2 = Background
# 3 = Boundary
#
# Binary segmentation:
# 1 -> Foreground
# 2, 3 -> Background


# ============================================================
# 3. Model Configuration
# ============================================================

IN_CHANNELS = 3
OUT_CHANNELS = 1

# Original U-Net channel configuration
BASE_CHANNELS = 64

# U-Net encoder channels:
# 64 -> 128 -> 256 -> 512 -> 1024


# ============================================================
# 4. Training Configuration
# ============================================================

BATCH_SIZE = 2
EPOCHS = 100

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0

USE_AMP = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 5. Loss Configuration
# ============================================================

BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5


# ============================================================
# 6. Checkpoint Configuration
# ============================================================

BEST_MODEL_PATH = CHECKPOINT_DIR / "unet_baseline_best.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "unet_baseline_last.pth"

# Whether to resume from the last checkpoint
RESUME = True


# ============================================================
# 7. Evaluation Configuration
# ============================================================

THRESHOLD = 0.5

SAVE_PREDICTIONS = True
PREDICTION_DIR = RUN_DIR / "predictions"
