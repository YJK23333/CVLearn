
import math

import torch

import config


def generate_base_anchors(
    sizes=config.ANCHOR_SIZES,
    ratios=config.ANCHOR_RATIOS,
):

    anchors = []

    for size in sizes:

        area = size * size

        for ratio in ratios:

            width = math.sqrt(area / ratio)
            height = width * ratio

            anchors.append([
                -width / 2,
                -height / 2,
                width / 2,
                height / 2
            ])

    return torch.tensor(
        anchors,
        dtype=torch.float32
    )


def generate_anchors(
    feature_height,
    feature_width,
    stride=config.FEATURE_STRIDE,
    device="cpu",
):

    base_anchors = generate_base_anchors().to(device)

    shifts_x = (
        torch.arange(
            feature_width,
            device=device
        ) + 0.5
    ) * stride

    shifts_y = (
        torch.arange(
            feature_height,
            device=device
        ) + 0.5
    ) * stride

    shift_y, shift_x = torch.meshgrid(
        shifts_y,
        shifts_x,
        indexing="ij"
    )

    shifts = torch.stack(
        [
            shift_x.reshape(-1),
            shift_y.reshape(-1),
            shift_x.reshape(-1),
            shift_y.reshape(-1)
        ],
        dim=1
    )

    anchors = (
        shifts[:, None, :]
        + base_anchors[None, :, :]
    )

    return anchors.reshape(-1, 4)


def box_iou(boxes1, boxes2):

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        *
        (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        *
        (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    lt = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2]
    )

    rb = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:]
    )

    wh = (rb - lt).clamp(min=0)

    intersection = wh[:, :, 0] * wh[:, :, 1]

    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    )

    return intersection / union.clamp(min=1e-6)


def encode_boxes(anchors, gt_boxes):

    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]

    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]

    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    dx = (gt_ctr_x - ctr_x) / widths.clamp(min=1e-6)
    dy = (gt_ctr_y - ctr_y) / heights.clamp(min=1e-6)

    dw = torch.log(
        gt_widths.clamp(min=1e-6)
        / widths.clamp(min=1e-6)
    )

    dh = torch.log(
        gt_heights.clamp(min=1e-6)
        / heights.clamp(min=1e-6)
    )

    return torch.stack(
        [dx, dy, dw, dh],
        dim=1
    )


def decode_boxes(anchors, deltas):

    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]

    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx, dy, dw, dh = deltas.unbind(dim=1)

    dw = dw.clamp(max=math.log(1000.0 / 16))
    dh = dh.clamp(max=math.log(1000.0 / 16))

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y

    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    return torch.stack(
        [
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h
        ],
        dim=1
    )


def match_anchors(anchors, gt_boxes):

    num_anchors = anchors.shape[0]

    labels = torch.full(
        (num_anchors,),
        -1,
        dtype=torch.float32,
        device=anchors.device
    )

    if gt_boxes.numel() == 0:

        labels[:] = 0

        return labels, torch.zeros_like(anchors)

    ious = box_iou(anchors, gt_boxes)

    max_iou, matched_indices = ious.max(dim=1)

    labels[max_iou < config.RPN_NEGATIVE_IOU] = 0

    labels[max_iou >= config.RPN_POSITIVE_IOU] = 1

    best_anchor_per_gt = ious.argmax(dim=0)

    labels[best_anchor_per_gt] = 1

    matched_gt = gt_boxes[matched_indices]

    targets = encode_boxes(
        anchors,
        matched_gt
    )

    # Subsample positive and negative anchors.
    positive = torch.where(labels == 1)[0]
    negative = torch.where(labels == 0)[0]

    max_positive = int(
        config.RPN_BATCH_SIZE
        * config.RPN_POSITIVE_FRACTION
    )

    if positive.numel() > max_positive:

        disable = positive[
            torch.randperm(
                positive.numel(),
                device=anchors.device
            )[max_positive:]
        ]

        labels[disable] = -1

    num_positive = (labels == 1).sum().item()

    max_negative = config.RPN_BATCH_SIZE - num_positive

    negative = torch.where(labels == 0)[0]

    if negative.numel() > max_negative:

        disable = negative[
            torch.randperm(
                negative.numel(),
                device=anchors.device
            )[max_negative:]
        ]

        labels[disable] = -1

    return labels, targets
