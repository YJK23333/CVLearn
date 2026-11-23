
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import nms

import config

from anchor import (
    generate_anchors,
    match_anchors,
    decode_boxes,
)


class RPN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv2d(
            config.FEATURE_CHANNELS,
            config.RPN_HIDDEN_CHANNELS,
            kernel_size=3,
            padding=1
        )

        self.objectness = nn.Conv2d(
            config.RPN_HIDDEN_CHANNELS,
            config.NUM_ANCHORS,
            kernel_size=1
        )

        self.bbox_pred = nn.Conv2d(
            config.RPN_HIDDEN_CHANNELS,
            config.NUM_ANCHORS * 4,
            kernel_size=1
        )

    def forward(
        self,
        feature_map,
        image_size,
        gt_boxes=None
    ):

        batch_size = feature_map.shape[0]

        x = F.relu(self.conv(feature_map))

        objectness = self.objectness(x)

        bbox_deltas = self.bbox_pred(x)

        # [B, A, H, W] -> [B, H*W*A]
        objectness = objectness.permute(
            0, 2, 3, 1
        ).reshape(batch_size, -1)

        # [B, A*4, H, W] -> [B, H*W*A, 4]
        bbox_deltas = bbox_deltas.permute(
            0, 2, 3, 1
        ).reshape(batch_size, -1, 4)

        _, _, feature_height, feature_width = feature_map.shape

        anchors = generate_anchors(
            feature_height,
            feature_width,
            device=feature_map.device
        )

        proposals = []

        for batch_idx in range(batch_size):

            scores = torch.sigmoid(
                objectness[batch_idx]
            )

            deltas = bbox_deltas[batch_idx]

            boxes = decode_boxes(
                anchors,
                deltas
            )

            height, width = image_size

            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
                0, width
            )

            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
                0, height
            )

            box_width = boxes[:, 2] - boxes[:, 0]
            box_height = boxes[:, 3] - boxes[:, 1]

            valid = (
                (box_width >= config.RPN_MIN_SIZE)
                &
                (box_height >= config.RPN_MIN_SIZE)
            )

            boxes = boxes[valid]
            scores = scores[valid]

            if self.training:

                pre_nms_topk = config.RPN_PRE_NMS_TOPK_TRAIN
                post_nms_topk = config.RPN_POST_NMS_TOPK_TRAIN

            else:

                pre_nms_topk = config.RPN_PRE_NMS_TOPK_TEST
                post_nms_topk = config.RPN_POST_NMS_TOPK_TEST

            num_topk = min(
                pre_nms_topk,
                scores.numel()
            )

            top_scores, top_indices = scores.topk(
                num_topk
            )

            boxes = boxes[top_indices]

            keep = nms(
                boxes,
                top_scores,
                config.RPN_NMS_THRESH
            )

            keep = keep[:post_nms_topk]

            proposals.append(
                boxes[keep].detach()
            )

        rpn_targets = None

        if self.training and gt_boxes is not None:

            labels, bbox_targets = match_anchors(
                anchors,
                gt_boxes[0]
            )

            rpn_targets = {
                "labels": labels,
                "bbox_targets": bbox_targets,
            }

        return proposals, objectness, bbox_deltas, rpn_targets
