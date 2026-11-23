
import torch
import torch.nn as nn

from torchvision.ops import roi_pool

import config


class ROIPool(nn.Module):

    def __init__(
        self,
        output_size=config.ROI_OUTPUT_SIZE
    ):

        super().__init__()

        self.output_size = output_size

    def forward(
        self,
        feature_map,
        proposals
    ):

        rois = []

        for batch_idx, boxes in enumerate(proposals):

            if boxes.numel() == 0:
                continue

            batch_indices = torch.full(
                (boxes.shape[0], 1),
                batch_idx,
                dtype=boxes.dtype,
                device=boxes.device
            )

            rois.append(
                torch.cat(
                    [batch_indices, boxes],
                    dim=1
                )
            )

        if len(rois) == 0:

            return feature_map.new_zeros(
                (0, feature_map.shape[1],
                 self.output_size, self.output_size)
            )

        rois = torch.cat(rois, dim=0)

        pooled = roi_pool(
            feature_map,
            rois,
            output_size=self.output_size,
            spatial_scale=1.0 / config.FEATURE_STRIDE
        )

        return pooled
