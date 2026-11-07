import torch
import torch.nn as nn

from torchvision.models import (
    vgg16,
    VGG16_Weights,
)

import config
from roi_pool import ROIPool


class FastRCNN(nn.Module):

    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
    ):

        super().__init__()

        if pretrained:

            weights = VGG16_Weights.IMAGENET1K_V1

        else:

            weights = None

        backbone = vgg16(
            weights=weights
        )

        # VGG16 Conv5 output.
        # Remove the final MaxPool layer.
        self.features = nn.Sequential(
            *list(backbone.features.children())[:-1]
        )

        # Reuse pretrained FC6 and FC7.
        self.fc6 = backbone.classifier[0]

        self.fc7 = backbone.classifier[3]

        if config.FREEZE_BACKBONE:

            for parameter in self.features.parameters():

                parameter.requires_grad = False

        self.roi_pool = ROIPool(
            output_size=config.ROI_SIZE
        )

        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(
            p=0.5
        )

        self.cls_score = nn.Linear(
            config.FC_DIM,
            num_classes,
        )

        self.bbox_pred = nn.Linear(
            config.FC_DIM,
            4 * (num_classes - 1),
        )

        nn.init.normal_(
            self.cls_score.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.constant_(
            self.cls_score.bias,
            0,
        )

        nn.init.normal_(
            self.bbox_pred.weight,
            mean=0.0,
            std=0.001,
        )

        nn.init.constant_(
            self.bbox_pred.bias,
            0,
        )

    def forward(
        self,
        images,
        proposals,
    ):
        """
        images:
            list of [3, H, W]

        proposals:
            list of [N, 4]

        Returns:
            list of prediction dictionaries
        """

        outputs = []

        for image, image_proposals in zip(
            images,
            proposals,
        ):

            # Extract shared convolutional feature map.
            feature_map = self.features(
                image.unsqueeze(0)
            )

            roi_features = []

            # Process RoIs in chunks to control GPU memory.
            for start in range(
                0,
                len(image_proposals),
                config.ROI_CHUNK_SIZE,
            ):

                end = min(
                    start + config.ROI_CHUNK_SIZE,
                    len(image_proposals),
                )

                current_rois = image_proposals[
                    start:end
                ].to(feature_map.device)

                pooled = self.roi_pool(
                    feature_map,
                    current_rois,
                )

                x = pooled.flatten(
                    start_dim=1
                )

                x = self.relu(
                    self.fc6(x)
                )

                x = self.dropout(x)

                x = self.relu(
                    self.fc7(x)
                )

                x = self.dropout(x)

                roi_features.append(x)

            if not roi_features:

                outputs.append(
                    {
                        "class_logits": None,
                        "bbox_deltas": None,
                    }
                )

                continue

            roi_features = torch.cat(
                roi_features,
                dim=0,
            )

            class_logits = self.cls_score(
                roi_features
            )

            bbox_deltas = self.bbox_pred(
                roi_features
            )

            outputs.append(
                {
                    "class_logits": class_logits,
                    "bbox_deltas": bbox_deltas,
                }
            )

        return outputs