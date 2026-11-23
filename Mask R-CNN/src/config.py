
import os
import random

import numpy as np
import torch


# ============================================================
# 1. Project Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_ROOT = (
    r"E:\3.Datas\Github\YJK23333\CVLearn"
    r"\data\oxford-iiit-pet"
)

IMAGE_DIR = os.path.join(DATASET_ROOT, "images")

TRIMAP_DIR = os.path.join(
    DATASET_ROOT,
    "annotations",
    "trimaps"
)

TRAINVAL_FILE = os.path.join(
    DATASET_ROOT,
    "annotations",
    "trainval.txt"
)

TEST_FILE = os.path.join(
    DATASET_ROOT,
    "annotations",
    "test.txt"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)


# ============================================================
# 2. Dataset Configuration
# ============================================================

SEED = 42

VAL_RATIO = 0.15

NUM_CLASSES = 3

CLASS_NAMES = {
    0: "background",
    1: "cat",
    2: "dog",
}

# Oxford Trimap:
# 1 = Pet
# 2 = Background
# 3 = Boundary

FOREGROUND_VALUES = (1, 3)


# ============================================================
# 3. Model Configuration
# ============================================================

PRETRAINED = True

IMAGE_MIN_SIZE = 400

IMAGE_MAX_SIZE = 640

RPN_PRE_NMS_TOP_N_TRAIN = 2000
RPN_PRE_NMS_TOP_N_TEST = 1000

RPN_POST_NMS_TOP_N_TRAIN = 1000
RPN_POST_NMS_TOP_N_TEST = 500

RPN_NMS_THRESH = 0.7

BOX_BATCH_SIZE_PER_IMAGE = 128

BOX_POSITIVE_FRACTION = 0.25

BOX_DETECTIONS_PER_IMAGE = 100

BOX_NMS_THRESH = 0.5

BOX_SCORE_THRESH = 0.05

MASK_THRESHOLD = 0.5


# ============================================================
# 4. Training Configuration
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

EPOCHS = 20

BATCH_SIZE = 1

NUM_WORKERS = 2

PIN_MEMORY = True

AMP = True

LEARNING_RATE = 0.005

MOMENTUM = 0.9

WEIGHT_DECAY = 0.0005

SCHEDULER_PATIENCE = 3

SCHEDULER_FACTOR = 0.5

MIN_LEARNING_RATE = 1e-6

GRAD_CLIP_NORM = 5.0


# ============================================================
# 5. Evaluation Configuration
# ============================================================

SCORE_THRESHOLD = 0.5

IOU_THRESHOLD = 0.5

MAX_VISUALIZE_IMAGES = 30


# ============================================================
# 6. Checkpoint Configuration
# ============================================================

LAST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "mask_rcnn_last.pth"
)

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "mask_rcnn_best.pth"
)


# ============================================================
# 7. Random Seed
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# 8. Directory Initialization
# ============================================================

def create_directories():

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
