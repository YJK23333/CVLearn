import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class FastRCNNLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.cls_loss = nn.CrossEntropyLoss(
            reduction="mean"
        )

    def forward(
        self,
        output,
        labels,
        bbox_targets,
    ):

        logits = output["class_logits"]

        bbox_deltas = output["bbox_deltas"]

        device = logits.device

        labels = labels.to(device)

        bbox_targets = bbox_targets.to(device)

        # Classification loss.
        classification_loss = self.cls_loss(
            logits,
            labels,
        )

        # Only positive RoIs participate in bbox regression.
        positive = labels > 0

        if positive.any():

            positive_labels = labels[positive]

            positive_deltas = bbox_deltas[
                positive
            ].reshape(
                -1,
                config.BBOX_REG_CLASSES,
                4,
            )

            # Convert class IDs from [1, 20] to [0, 19].
            class_indices = (
                positive_labels - 1
            )

            row_indices = torch.arange(
                len(class_indices),
                device=device,
            )

            positive_deltas = positive_deltas[
                row_indices,
                class_indices,
            ]

            positive_targets = bbox_targets[
                positive
            ]

            regression_loss = F.smooth_l1_loss(
                positive_deltas,
                positive_targets,
                reduction="sum",
            ) / max(len(labels), 1)

        else:

            regression_loss = bbox_deltas.sum() * 0.0

        total_loss = (
            classification_loss
            +
            config.BBOX_LOSS_WEIGHT
            * regression_loss
        )

        return (
            total_loss,
            classification_loss,
            regression_loss,
        )