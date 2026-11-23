
from pathlib import Path

import torch


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(
    r"E:\3.Datas\Github\YJK23333\CVLearn\data\voc2007"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Dataset
# ============================================================

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

NUM_CLASSES = len(VOC_CLASSES) + 1

CLASS_TO_IDX = {
    name: index + 1
    for index, name in enumerate(VOC_CLASSES)
}

IDX_TO_CLASS = {
    index: name
    for name, index in CLASS_TO_IDX.items()
}

BACKGROUND_IDX = 0


# ============================================================
# Image Transform
# ============================================================

MIN_SIZE = 600
MAX_SIZE = 1000

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


# ============================================================
# Backbone
# ============================================================

PRETRAINED = True

FEATURE_CHANNELS = 512
FEATURE_STRIDE = 16


# ============================================================
# Anchor
# ============================================================

ANCHOR_SIZES = [128, 256, 512]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]

NUM_ANCHORS = len(ANCHOR_SIZES) * len(ANCHOR_RATIOS)


# ============================================================
# RPN
# ============================================================

RPN_HIDDEN_CHANNELS = 512

RPN_PRE_NMS_TOPK_TRAIN = 6000
RPN_POST_NMS_TOPK_TRAIN = 1000

RPN_PRE_NMS_TOPK_TEST = 3000
RPN_POST_NMS_TOPK_TEST = 300

RPN_NMS_THRESH = 0.7

RPN_POSITIVE_IOU = 0.7
RPN_NEGATIVE_IOU = 0.3

RPN_BATCH_SIZE = 256
RPN_POSITIVE_FRACTION = 0.5

RPN_MIN_SIZE = 16


# ============================================================
# RoI
# ============================================================

ROI_OUTPUT_SIZE = 7
ROI_SAMPLING_RATIO = 2

ROI_BATCH_SIZE = 128
ROI_POSITIVE_FRACTION = 0.25

ROI_POSITIVE_IOU = 0.5

ROI_NMS_THRESH = 0.3
ROI_SCORE_THRESH = 0.05
ROI_MAX_DETECTIONS = 100


# ============================================================
# BBox Regression
# ============================================================

BBOX_NORMALIZE_MEAN = [0.0, 0.0, 0.0, 0.0]
BBOX_NORMALIZE_STD = [0.1, 0.1, 0.2, 0.2]


# ============================================================
# Training
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 20
BATCH_SIZE = 1
NUM_WORKERS = 2

LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

LR_STEP_SIZE = 8
LR_GAMMA = 0.1

AMP = True
GRAD_CLIP_NORM = 10.0

SEED = 42


# ============================================================
# Loss
# ============================================================

RPN_BBOX_LOSS_WEIGHT = 1.0
ROI_BBOX_LOSS_WEIGHT = 1.0


# ============================================================
# Checkpoints
# ============================================================

BEST_CHECKPOINT = CHECKPOINT_DIR / "faster_rcnn_best.pth"
LAST_CHECKPOINT = CHECKPOINT_DIR / "faster_rcnn_last.pth"


# ============================================================
# Evaluation
# ============================================================

EVAL_IOU_THRESH = 0.5
VOC_AP_MODE = "11point"


# ============================================================
# Logging
# ============================================================

PRINT_FREQ = 50
SAVE_FREQ = 1
