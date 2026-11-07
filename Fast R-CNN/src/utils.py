import random

import numpy as np
import torch
import torch.nn.functional as F

from torchvision.ops import nms

import config


def set_seed(seed=config.RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_box(proposals, deltas):

    px = (
        proposals[:, 0] + proposals[:, 2]
    ) * 0.5

    py = (
        proposals[:, 1] + proposals[:, 3]
    ) * 0.5

    pw = (
        proposals[:, 2] - proposals[:, 0] + 1
    )

    ph = (
        proposals[:, 3] - proposals[:, 1] + 1
    )

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2].clamp(max=4.135)
    dh = deltas[:, 3].clamp(max=4.135)

    gx = dx * pw + px
    gy = dy * ph + py

    gw = torch.exp(dw) * pw
    gh = torch.exp(dh) * ph

    x1 = gx - 0.5 * gw
    y1 = gy - 0.5 * gh

    x2 = gx + 0.5 * gw - 1
    y2 = gy + 0.5 * gh - 1

    return torch.stack(
        [x1, y1, x2, y2],
        dim=1,
    )


def clip_boxes(boxes, image_size):

    image_width, image_height = image_size

    boxes[:, 0].clamp_(
        0,
        image_width - 1,
    )

    boxes[:, 1].clamp_(
        0,
        image_height - 1,
    )

    boxes[:, 2].clamp_(
        0,
        image_width - 1,
    )

    boxes[:, 3].clamp_(
        0,
        image_height - 1,
    )

    return boxes


def postprocess_detections(
    class_logits,
    bbox_deltas,
    proposals,
    image_size,
):

    probabilities = F.softmax(
        class_logits.float(),
        dim=1,
    )

    bbox_deltas = bbox_deltas.float()

    detections = []

    for class_id in range(
        1,
        config.NUM_CLASSES,
    ):

        scores = probabilities[:, class_id]

        keep = (
            scores >= config.CONFIDENCE_THRESHOLD
        )

        if not keep.any():
            continue

        class_proposals = proposals[keep]
        class_scores = scores[keep]

        class_deltas = bbox_deltas[
            keep,
            (class_id - 1) * 4:
            (class_id - 1) * 4 + 4,
        ]

        boxes = decode_box(
            class_proposals,
            class_deltas,
        )

        boxes = clip_boxes(
            boxes,
            image_size,
        )

        valid = (
            (boxes[:, 2] >= boxes[:, 0])
            &
            (boxes[:, 3] >= boxes[:, 1])
        )

        boxes = boxes[valid]
        class_scores = class_scores[valid]

        if len(boxes) == 0:
            continue

        # torchvision NMS uses continuous coordinates.
        # Convert inclusive coordinates to continuous coordinates
        # by adding 1 to x2 and y2.
        nms_boxes = boxes.clone()
        nms_boxes[:, 2:] += 1

        keep_indices = nms(
            nms_boxes,
            class_scores,
            config.NMS_THRESHOLD,
        )

        for idx in keep_indices:

            detections.append(
                {
                    "box": boxes[idx].detach().cpu(),
                    "score": class_scores[idx].item(),
                    "label": class_id,
                }
            )

    detections.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return detections[
        :config.MAX_DETECTIONS_PER_IMAGE
    ]


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_metric,
    path,
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(
    model,
    path,
    optimizer=None,
    scheduler=None,
):

    checkpoint = torch.load(
        path,
        map_location=config.DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return checkpoint