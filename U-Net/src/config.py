from pathlib import Path
import torch

SRC_DIR = Path(__file__).resolve().parent
UNET_DIR = SRC_DIR.parent
PROJECT_ROOT = UNET_DIR.parent

DATA_ROOT = PROJECT_ROOT / "data"
DATASET_ROOT = DATA_ROOT / "oxford-iiit-pet"
CHECKPOINT_DIR = UNET_DIR / "checkpoints"
OUTPUT_DIR = UNET_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# ============================================================
# Dataset configuration
# ============================================================
DATASET_NAME = "Oxford-IIIT Pet"
TRAINVAL_SPLIT = "trainval"
OFFICIAL_TEST_SPLIT = "test"
VALIDATION_RATIO = 0.15
DOWNLOAD_DATASET = True

# Oxford-IIIT Pet trimap values:
# 1 = pet, 2 = background, 3 = uncertain boundary.
# For this starter binary task, boundary pixels are treated as pet.
PET_MASK_VALUES = (1, 3)
BACKGROUND_MASK_VALUE = 2

# ============================================================
# Input / output configuration
# ============================================================
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_CHANNELS = 3
NUM_CLASSES = 1  # binary segmentation: one logit channel

# ============================================================
# U-Net architecture configuration
# ============================================================
BASE_CHANNELS = 32
CHANNEL_MULTIPLIERS = (1, 2, 4, 8, 16)

CONV_KERNEL_SIZE = 3
CONV_PADDING = 1
CONV_STRIDE = 1
CONV_LAYERS_PER_BLOCK = 2
USE_BATCH_NORM = False
USE_CONV_BIAS = True
ACTIVATION_INPLACE = True

POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2

UPSAMPLE_KERNEL_SIZE = 2
UPSAMPLE_STRIDE = 2

OUTPUT_KERNEL_SIZE = 1
OUTPUT_STRIDE = 1
OUTPUT_PADDING = 0

# ============================================================
# Training configuration
# ============================================================
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0  # safe default for Windows
PIN_MEMORY = torch.cuda.is_available()
RANDOM_SEED = 42
USE_AMP = False  # can enable later if using a compatible CUDA setup

# ============================================================
# Augmentation and evaluation
# ============================================================
USE_HORIZONTAL_FLIP = True
HORIZONTAL_FLIP_PROBABILITY = 0.5
THRESHOLD = 0.5
SAVE_PREDICTION_LIMIT = 12

# Automatically select CUDA when available, otherwise CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
