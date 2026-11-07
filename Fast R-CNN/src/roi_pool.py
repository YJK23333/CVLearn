import torch
import torch.nn as nn

from torchvision.ops import roi_pool

import config


class ROIPool(nn.Module):

    def __init__(self, output_size=7):

        super().__init__()

        self.output_size = output_size

    def forward(
        self,
        feature_map,
        rois,
    ):
        """
        feature_map:
            [1, C, H, W]

        rois:
            [N, 4]
            x1, y1, x2, y2

        output:
            [N, C, 7, 7]
        """

        if len(rois) == 0:

            return feature_map.new_zeros(
                (
                    0,
                    feature_map.shape[1],
                    self.output_size,
                    self.output_size,
                )
            )

        batch_indices = torch.zeros(
            (len(rois), 1),
            dtype=rois.dtype,
            device=rois.device,
        )

        rois_with_batch = torch.cat(
            [
                batch_indices,
                rois,
            ],
            dim=1,
        )

        pooled = roi_pool(
            feature_map,
            rois_with_batch,
            output_size=self.output_size,
            spatial_scale=1.0 / config.FEATURE_STRIDE,
        )

        return pooled