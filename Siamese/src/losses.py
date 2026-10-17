import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, e1, e2, label):
        d = F.pairwise_distance(e1, e2, p=2)

        pos = label * d.pow(2)
        neg = (1 - label) * F.relu(self.margin - d).pow(2)
        loss = 0.5 * (pos + neg)
        return loss.mean()

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = F.pairwise_distance(anchor, positive, p=2)
        d_neg = F.pairwise_distance(anchor, negative, p=2)
        return F.relu(d_pos - d_neg + self.margin).mean()
