
import torch
import torch.nn as nn

from backbone import VGG16Backbone
from rpn import RPN
from roi_pool import ROIPool

import config


class DetectionHead(nn.Module):

    def __init__(self):

        super().__init__()

        input_channels = (
            config.FEATURE_CHANNELS
            * config.ROI_OUTPUT_SIZE
            * config.ROI_OUTPUT_SIZE
        )

        self.fc1 = nn.Linear(
            input_channels,
            4096
        )

        self.fc2 = nn.Linear(
            4096,
            4096
        )

        self.cls_score = nn.Linear(
            4096,
            config.NUM_CLASSES
        )

        self.bbox_pred = nn.Linear(
            4096,
            config.NUM_CLASSES * 4
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = x.flatten(start_dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        cls_logits = self.cls_score(x)

        bbox_deltas = self.bbox_pred(x)

        return cls_logits, bbox_deltas


class FasterRCNN(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        self.backbone = VGG16Backbone(
            pretrained=pretrained
        )

        self.rpn = RPN()

        self.roi_pool = ROIPool()

        self.head = DetectionHead()

    def forward(
        self,
        images,
        targets=None
    ):

        if isinstance(images, list):
            images = torch.stack(images)

        image_size = images.shape[-2:]

        feature_map = self.backbone(images)

        gt_boxes = None

        if targets is not None:

            gt_boxes = [
                target["boxes"]
                for target in targets
            ]

        proposals, objectness, rpn_bbox_deltas, rpn_targets = self.rpn(
            feature_map,
            image_size,
            gt_boxes
        )

        cls_logits = None
        roi_bbox_deltas = None

        # During training, Detection Head is evaluated
        # only on sampled RoIs inside loss.py.
        if not self.training:

            pooled_features = self.roi_pool(
                feature_map,
                proposals
            )

            cls_logits, roi_bbox_deltas = self.head(
                pooled_features
            )

        return {
            "proposals": proposals,
            "objectness": objectness,
            "rpn_bbox_deltas": rpn_bbox_deltas,
            "rpn_targets": rpn_targets,
            "cls_logits": cls_logits,
            "roi_bbox_deltas": roi_bbox_deltas,
            "feature_map": feature_map,
            "model_head": self.head,
        }
