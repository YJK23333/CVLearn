
import random

import numpy as np
import torch

from torchvision.ops import nms

import config

from anchor import decode_boxes


def set_seed(seed=config.SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():

    return torch.device(config.DEVICE)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_loss,
    path
):

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        },
        path
    )


def load_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    path=None
):

    if path is None or not path.exists():

        return 0, float("inf")

    checkpoint = torch.load(
        path,
        map_location=config.DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

    return (
        checkpoint["epoch"] + 1,
        checkpoint["best_loss"]
    )


@torch.no_grad()
def postprocess_detections(
    outputs,
    image_size
):

    proposals = outputs["proposals"][0]

    cls_logits = outputs["cls_logits"]

    bbox_deltas = outputs["roi_bbox_deltas"]

    if proposals.numel() == 0:

        return {
            "boxes": torch.zeros(
                (0, 4),
                device=cls_logits.device
            ),
            "scores": torch.zeros(
                (0,),
                device=cls_logits.device
            ),
            "labels": torch.zeros(
                (0,),
                dtype=torch.long,
                device=cls_logits.device
            )
        }

    scores = torch.softmax(
        cls_logits,
        dim=1
    )

    bbox_deltas = bbox_deltas.reshape(
        -1,
        config.NUM_CLASSES,
        4
    )

    all_boxes = []
    all_scores = []
    all_labels = []

    height, width = image_size

    for class_idx in range(1, config.NUM_CLASSES):

        class_scores = scores[:, class_idx]

        keep = class_scores > config.ROI_SCORE_THRESH

        if not keep.any():
            continue

        selected_proposals = proposals[keep]

        selected_deltas = bbox_deltas[
            keep,
            class_idx
        ]

        selected_scores = class_scores[keep]

        decoded = decode_boxes(
            selected_proposals,
            selected_deltas
        )

        decoded[:, [0, 2]] = decoded[:, [0, 2]].clamp(
            0, width
        )

        decoded[:, [1, 3]] = decoded[:, [1, 3]].clamp(
            0, height
        )

        valid = (
            (decoded[:, 2] > decoded[:, 0])
            &
            (decoded[:, 3] > decoded[:, 1])
        )

        decoded = decoded[valid]
        selected_scores = selected_scores[valid]

        if decoded.numel() == 0:
            continue

        keep = nms(
            decoded,
            selected_scores,
            config.ROI_NMS_THRESH
        )

        decoded = decoded[keep]
        selected_scores = selected_scores[keep]

        all_boxes.append(decoded)
        all_scores.append(selected_scores)

        all_labels.append(
            torch.full(
                (decoded.shape[0],),
                class_idx,
                dtype=torch.long,
                device=decoded.device
            )
        )

    if len(all_boxes) == 0:

        return {
            "boxes": torch.zeros(
                (0, 4),
                device=cls_logits.device
            ),
            "scores": torch.zeros(
                (0,),
                device=cls_logits.device
            ),
            "labels": torch.zeros(
                (0,),
                dtype=torch.long,
                device=cls_logits.device
            )
        }

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    keep = torch.argsort(
        scores,
        descending=True
    )[:config.ROI_MAX_DETECTIONS]

    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep]
    }


def voc_ap(rec, prec):

    if config.VOC_AP_MODE == "11point":

        ap = 0.0

        for threshold in np.arange(
            0.0,
            1.1,
            0.1
        ):

            if np.sum(rec >= threshold) == 0:
                precision = 0
            else:
                precision = np.max(
                    prec[rec >= threshold]
                )

            ap += precision / 11.0

        return ap

    mrec = np.concatenate(
        ([0.0], rec, [1.0])
    )

    mpre = np.concatenate(
        ([0.0], prec, [0.0])
    )

    for i in range(
        len(mpre) - 1,
        0,
        -1
    ):

        mpre[i - 1] = max(
            mpre[i - 1],
            mpre[i]
        )

    indices = np.where(
        mrec[1:] != mrec[:-1]
    )[0]

    return np.sum(
        (mrec[indices + 1] - mrec[indices])
        * mpre[indices + 1]
    )


def evaluate_voc(
    predictions,
    ground_truths
):

    ap_per_class = {}

    for class_idx, class_name in enumerate(
        config.VOC_CLASSES,
        start=1
    ):

        class_predictions = []

        total_gt = 0

        matched = {}

        for image_id, gt in ground_truths.items():

            gt_mask = gt["labels"] == class_idx

            boxes = gt["boxes"][gt_mask]

            matched[image_id] = np.zeros(
                boxes.shape[0],
                dtype=bool
            )

            total_gt += boxes.shape[0]

        for image_id, pred in predictions.items():

            mask = pred["labels"] == class_idx

            boxes = pred["boxes"][mask]
            scores = pred["scores"][mask]

            for box, score in zip(boxes, scores):

                class_predictions.append(
                    (
                        image_id,
                        float(score),
                        box
                    )
                )

        class_predictions.sort(
            key=lambda x: x[1],
            reverse=True
        )

        if total_gt == 0:

            ap_per_class[class_name] = 0.0
            continue

        tp = np.zeros(
            len(class_predictions)
        )

        fp = np.zeros(
            len(class_predictions)
        )

        for i, (
            image_id,
            score,
            box
        ) in enumerate(class_predictions):

            gt_boxes = ground_truths[
                image_id
            ]["boxes"][
                ground_truths[image_id]["labels"]
                == class_idx
            ]

            if gt_boxes.numel() == 0:

                fp[i] = 1
                continue

            box = box.reshape(1, 4)

            from anchor import box_iou

            ious = box_iou(
                box,
                gt_boxes
            )[0]

            best_idx = torch.argmax(ious).item()

            best_iou = ious[best_idx].item()

            if best_iou >= config.EVAL_IOU_THRESH:

                if not matched[image_id][best_idx]:

                    tp[i] = 1
                    matched[image_id][best_idx] = True

                else:

                    fp[i] = 1

            else:

                fp[i] = 1

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        rec = tp / total_gt

        prec = tp / np.maximum(
            tp + fp,
            np.finfo(np.float64).eps
        )

        ap_per_class[class_name] = voc_ap(
            rec,
            prec
        )

    mean_ap = np.mean(
        list(ap_per_class.values())
    )

    return ap_per_class, mean_ap
