from pathlib import Path
import torch


# ============================================================
# Project paths
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_ROOT = PROJECT_ROOT.parent / "data"
VOC_ROOT = DATA_ROOT / "voc2007"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PROPOSAL_CACHE_DIR = OUTPUT_DIR / "proposals"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROPOSAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Dataset
# ============================================================

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

NUM_CLASSES = len(VOC_CLASSES)

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(VOC_CLASSES)
}


# ============================================================
# Image preprocessing
# ============================================================

IMAGE_MIN_SIZE = 600
IMAGE_MAX_SIZE = 1000

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


# ============================================================
# Selective Search
# ============================================================

SELECTIVE_SEARCH_SCALE = 500
SELECTIVE_SEARCH_SIGMA = 0.9
SELECTIVE_SEARCH_MIN_SIZE = 10

MAX_PROPOSALS = 2000


# ============================================================
# Proposal sampling
# ============================================================

POSITIVE_IOU_THRESHOLD = 0.5
NEGATIVE_IOU_THRESHOLD = 0.1

MAX_POSITIVE_ROIS = 32
MAX_NEGATIVE_ROIS = 96

MAX_ROIS_PER_IMAGE = (
    MAX_POSITIVE_ROIS + MAX_NEGATIVE_ROIS
)


# ============================================================
# Backbone
# ============================================================

BACKBONE = "vgg16"

PRETRAINED = True

FREEZE_BACKBONE = True


# ============================================================
# RoI Pooling
# ============================================================

ROI_SIZE = 7

FEATURE_STRIDE = 16

ROI_CHUNK_SIZE = 128


# ============================================================
# Fast R-CNN Head
# ============================================================

FC_DIM = 4096

BBOX_REG_CLASSES = NUM_CLASSES - 1


# ============================================================
# Training
# ============================================================

BATCH_SIZE = 1

NUM_WORKERS = 0

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 5e-4

MOMENTUM = 0.9

LR_STEP_SIZE = 10

LR_GAMMA = 0.1

GRAD_CLIP_NORM = 10.0


# ============================================================
# Loss
# ============================================================

BBOX_LOSS_WEIGHT = 1.0


# ============================================================
# Evaluation
# ============================================================

IOU_THRESHOLD = 0.5

CONFIDENCE_THRESHOLD = 0.05

NMS_THRESHOLD = 0.3

MAX_DETECTIONS_PER_IMAGE = 100


# ============================================================
# Runtime
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

AMP = torch.cuda.is_available()

RANDOM_SEED = 42


# ============================================================
# Checkpoint
# ============================================================

CHECKPOINT_NAME = "fast_rcnn_last.pth"

BEST_CHECKPOINT_NAME = "fast_rcnn_best.pth"