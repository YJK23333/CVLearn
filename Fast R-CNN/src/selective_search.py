from pathlib import Path

import numpy as np
from PIL import Image
import selectivesearch

import config


def generate_proposals(image_path):
    """
    Generate region proposals using Selective Search.

    Returns:
        proposals: [N, 4]
        Format: x1, y1, x2, y2
    """

    image = np.asarray(
        Image.open(image_path).convert("RGB")
    )

    image_height, image_width = image.shape[:2]

    _, regions = selectivesearch.selective_search(
        image,
        scale=config.SELECTIVE_SEARCH_SCALE,
        sigma=config.SELECTIVE_SEARCH_SIGMA,
        min_size=config.SELECTIVE_SEARCH_MIN_SIZE,
    )

    proposals = []

    for region in regions:
        x, y, w, h = region["rect"]

        if w <= 1 or h <= 1:
            continue

        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(image_width - 1, x + w - 1)
        y2 = min(image_height - 1, y + h - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        proposals.append(
            [x1, y1, x2, y2]
        )

    if not proposals:
        proposals = [
            [
                0,
                0,
                image_width - 1,
                image_height - 1,
            ]
        ]

    proposals = np.asarray(
        proposals,
        dtype=np.float32,
    )

    proposals = remove_duplicate_boxes(proposals)

    if len(proposals) > config.MAX_PROPOSALS:
        proposals = proposals[:config.MAX_PROPOSALS]

    return proposals


def remove_duplicate_boxes(boxes):
    boxes = np.unique(
        boxes.astype(np.int32),
        axis=0,
    )

    return boxes.astype(np.float32)


def get_cache_path(image_path):
    image_id = Path(image_path).stem

    return (
        config.PROPOSAL_CACHE_DIR
        / f"{image_id}.npy"
    )


def load_or_generate_proposals(image_path):
    cache_path = get_cache_path(image_path)

    if cache_path.exists():
        return np.load(cache_path)

    proposals = generate_proposals(image_path)

    np.save(
        cache_path,
        proposals,
    )

    return proposals