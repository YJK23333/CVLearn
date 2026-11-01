
import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (probs * targets).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5
    ):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )

        return loss


@torch.no_grad()
def calculate_metrics(logits, targets, threshold=0.5):

    probs = torch.sigmoid(logits)

    preds = (probs >= threshold).long()
    targets = targets.long()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum().float()

    pred_sum = preds.sum().float()
    target_sum = targets.sum().float()

    union = pred_sum + target_sum - intersection

    total = targets.numel()

    correct = (preds == targets).sum().float()

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        pred_sum + target_sum + 1e-6
    )

    iou = (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )

    pixel_accuracy = correct / total

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "pixel_accuracy": pixel_accuracy.item()
    }


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_dice,
    path
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice
        },
        path
    )


def load_checkpoint(
    model,
    optimizer,
    path,
    device
):

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1
    best_dice = checkpoint["best_dice"]

    return model, optimizer, start_epoch, best_dice
