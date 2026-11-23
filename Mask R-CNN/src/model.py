
import torch
import torch.nn as nn

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

from torchvision.models.detection.mask_rcnn import (
    MaskRCNNPredictor
)

import config


# ============================================================
# 1. Build Mask R-CNN
# ============================================================

def build_model():

    print("Building Mask R-CNN...")

    weights = (
        MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        if config.PRETRAINED
        else None
    )

    model = maskrcnn_resnet50_fpn(
        weights=weights,

        min_size=config.IMAGE_MIN_SIZE,

        max_size=config.IMAGE_MAX_SIZE,

        rpn_pre_nms_top_n_train=(
            config.RPN_PRE_NMS_TOP_N_TRAIN
        ),

        rpn_pre_nms_top_n_test=(
            config.RPN_PRE_NMS_TOP_N_TEST
        ),

        rpn_post_nms_top_n_train=(
            config.RPN_POST_NMS_TOP_N_TRAIN
        ),

        rpn_post_nms_top_n_test=(
            config.RPN_POST_NMS_TOP_N_TEST
        ),

        rpn_nms_thresh=config.RPN_NMS_THRESH,

        box_batch_size_per_image=(
            config.BOX_BATCH_SIZE_PER_IMAGE
        ),

        box_positive_fraction=(
            config.BOX_POSITIVE_FRACTION
        ),

        box_detections_per_img=(
            config.BOX_DETECTIONS_PER_IMAGE
        ),

        box_nms_thresh=config.BOX_NMS_THRESH,

        box_score_thresh=config.BOX_SCORE_THRESH
    )

    # --------------------------------------------------------
    # Classification + Box Regression Head
    # --------------------------------------------------------

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        config.NUM_CLASSES
    )

    # --------------------------------------------------------
    # Mask Prediction Head
    # --------------------------------------------------------

    in_features_mask = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        config.NUM_CLASSES
    )

    print("Mask R-CNN initialized.")

    print(
        f"Number of classes: {config.NUM_CLASSES}"
    )

    return model


# ============================================================
# 2. Model Information
# ============================================================

def count_parameters(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 60)

    print(
        f"Total parameters: {total_params:,}"
    )

    print(
        f"Trainable parameters: {trainable_params:,}"
    )

    print("=" * 60)
