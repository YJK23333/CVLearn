import os
import torch


# ============================================================
# 1. Project Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.abspath(
    os.path.join(
        PROJECT_ROOT,
        "..",
        "data",
        "oxford-iiit-pet"
    )
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "deit"
)

TEACHER_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "teacher",
    "teacher_best.pth"
)


# ============================================================
# 2. Dataset
# ============================================================

NUM_CLASSES = 37

IMAGE_SIZE = 224

PATCH_SIZE = 16

VAL_SPLIT = 0.10

SEED = 42


# ============================================================
# 3. DeiT Model
# ============================================================

EMBED_DIM = 384

DEPTH = 6

NUM_HEADS = 6

MLP_RATIO = 4

DROPOUT = 0.1

ATTENTION_DROPOUT = 0.1


# ============================================================
# 4. Distillation
# ============================================================

DISTILL_ALPHA = 0.5


# ============================================================
# 5. Training
# ============================================================

BATCH_SIZE = 32

# Increased from 30 to 60
EPOCHS = 60

# Increased from 3e-4 to 5e-4
LEARNING_RATE = 5e-4

WEIGHT_DECAY = 0.05

NUM_WORKERS = 4

PIN_MEMORY = True

AMP = True


# ============================================================
# 6. Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 7. Checkpoints
# ============================================================

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "deit_best.pth"
)

LAST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "deit_last.pth"
)
