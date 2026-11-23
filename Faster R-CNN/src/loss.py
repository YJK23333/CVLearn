
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

from anchor import box_iou, encode_boxes


def smooth_l1_loss(pred, target):

    return F.smooth_l1_loss(
        pred,
        target,
        beta=1.0 / 9.0,
        reduction="sum"
    )


def rpn_loss(
    objectness,
    bbox_deltas,
    rpn_targets
):

    labels = rpn_targets["labels"]
    bbox_targets = rpn_targets["bbox_targets"]

    valid = labels >= 0
    positive = labels == 1

    if valid.any():

        cls_loss = F.binary_cross_entropy_with_logits(
            objectness[0][valid],
            labels[valid]
        )

    else:

        cls_loss = objectness.sum() * 0

    if positive.any():

        reg_loss = smooth_l1_loss(
            bbox_deltas[0][positive],
            bbox_targets[positive]
        ) / max(positive.sum().item(), 1)

    else:

        reg_loss = bbox_deltas.sum() * 0

    return cls_loss, reg_loss


def sample_roi_proposals(
    proposals,
    gt_boxes,
    gt_labels
):

    device = proposals.device

    # Append ground-truth boxes to proposals.
    proposals = torch.cat(
        [proposals, gt_boxes],
        dim=0
    )

    if gt_boxes.numel() == 0:

        count = min(
            proposals.shape[0],
            config.ROI_BATCH_SIZE
        )

        indices = torch.randperm(
            proposals.shape[0],
            device=device
        )[:count]

        return (
            proposals[indices],
            torch.zeros(
                count,
                dtype=torch.long,
                device=device
            ),
            torch.zeros(
                count,
                4,
                device=device
            )
        )

    ious = box_iou(
        proposals,
        gt_boxes
    )

    max_iou, matched_indices = ious.max(dim=1)

    matched_labels = gt_labels[matched_indices]

    labels = matched_labels.clone()

    labels[max_iou < config.ROI_POSITIVE_IOU] = 0

    positive = torch.where(labels > 0)[0]
    negative = torch.where(labels == 0)[0]

    num_positive = min(
        int(
            config.ROI_BATCH_SIZE
            * config.ROI_POSITIVE_FRACTION
        ),
        positive.numel()
    )

    num_negative = min(
        config.ROI_BATCH_SIZE - num_positive,
        negative.numel()
    )

    if positive.numel() > num_positive:

        positive = positive[
            torch.randperm(
                positive.numel(),
                device=device
            )[:num_positive]
        ]

    if negative.numel() > num_negative:

        negative = negative[
            torch.randperm(
                negative.numel(),
                device=device
            )[:num_negative]
        ]

    keep = torch.cat(
        [positive, negative]
    )

    sampled_proposals = proposals[keep]

    sampled_labels = labels[keep]

    bbox_targets = torch.zeros(
        (keep.numel(), 4),
        dtype=torch.float32,
        device=device
    )

    if positive.numel() > 0:

        matched_gt = gt_boxes[
            matched_indices[positive]
        ]

        bbox_targets[:positive.numel()] = encode_boxes(
            proposals[positive],
            matched_gt
        )

    return (
        sampled_proposals,
        sampled_labels,
        bbox_targets
    )


def roi_loss(
    cls_logits,
    bbox_deltas,
    sampled_labels,
    bbox_targets
):

    if sampled_labels.numel() == 0:

        zero = cls_logits.sum() * 0

        return zero, zero

    cls_loss = F.cross_entropy(
        cls_logits,
        sampled_labels
    )

    positive = torch.where(
        sampled_labels > 0
    )[0]

    if positive.numel() > 0:

        # Select the bbox regression output corresponding
        # to each positive proposal's ground-truth class.
        pred = bbox_deltas.reshape(
            -1,
            config.NUM_CLASSES,
            4
        )

        pred = pred[
            positive,
            sampled_labels[positive]
        ]

        reg_loss = smooth_l1_loss(
            pred,
            bbox_targets[positive]
        ) / max(positive.numel(), 1)

    else:

        reg_loss = bbox_deltas.sum() * 0

    return cls_loss, reg_loss


def compute_losses(outputs, targets):

    rpn_cls, rpn_reg = rpn_loss(
        outputs["objectness"],
        outputs["rpn_bbox_deltas"],
        outputs["rpn_targets"]
    )

    sampled_proposals, sampled_labels, bbox_targets = (
        sample_roi_proposals(
            outputs["proposals"][0],
            targets[0]["boxes"],
            targets[0]["labels"]
        )
    )

    # Recompute RoI features for sampled proposals.
    # This ensures the RoI head is trained using both
    # positive and negative proposals.
    roi = outputs["feature_map"]

    batch_indices = torch.zeros(
        (sampled_proposals.shape[0], 1),
        dtype=sampled_proposals.dtype,
        device=sampled_proposals.device
    )

    rois = torch.cat(
        [batch_indices, sampled_proposals],
        dim=1
    )

    from torchvision.ops import roi_pool

    pooled = roi_pool(
        roi,
        rois,
        output_size=config.ROI_OUTPUT_SIZE,
        spatial_scale=1.0 / config.FEATURE_STRIDE
    )

    cls_logits, bbox_deltas = outputs["model_head"](
        pooled
    )

    roi_cls, roi_reg = roi_loss(
        cls_logits,
        bbox_deltas,
        sampled_labels,
        bbox_targets
    )

    total_loss = (
        rpn_cls
        + config.RPN_BBOX_LOSS_WEIGHT * rpn_reg
        + roi_cls
        + config.ROI_BBOX_LOSS_WEIGHT * roi_reg
    )

    return {
        "total": total_loss,
        "rpn_cls": rpn_cls,
        "rpn_reg": rpn_reg,
        "roi_cls": roi_cls,
        "roi_reg": roi_reg,
    }
