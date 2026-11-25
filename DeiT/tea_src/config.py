
import os
import torch


# ============================================================
# 1. Project Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "data", "oxford-iiit-pet")
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT, "checkpoints", "teacher"
)


# ============================================================
# 2. Dataset
# ============================================================

NUM_CLASSES = 37

IMAGE_SIZE = 224

VAL_SPLIT = 0.10

SEED = 42


# ============================================================
# 3. Model
# ============================================================

MODEL_NAME = "resnet18"

PRETRAINED = True

DROPOUT = 0.0


# ============================================================
# 4. Training
# ============================================================

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

PIN_MEMORY = True

AMP = True


# ============================================================
# 5. Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 6. Checkpoints
# ============================================================

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "teacher_best.pth"
)

LAST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "teacher_last.pth"
)
