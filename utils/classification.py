import numpy as np
import cv2


def classify_image_from_masks(
    anomaly_mask: np.ndarray,
    foreground_mask: np.ndarray,
    max_anom_fg_ratio: float = 0.7,
    min_largest_cc_fg_ratio: float = 0.0050,
    min_largest_cc_pixels: int = 10,
) -> int:
    
    fg_area = int(foreground_mask.sum())
    if fg_area == 0:
        return 0

    # Anomaly restricted to foreground
    anom_in_fg = anomaly_mask & foreground_mask
    anom_fg_area = int(anom_in_fg.sum())

    if anom_fg_area == 0:
        # No anomaly on the object -> normal
        return 0

    # Rule 1: anomaly covers too much of the object
    anom_fg_ratio = anom_fg_area / fg_area
    if anom_fg_ratio >= max_anom_fg_ratio:
        return 0

    # Connected components on anomaly ∩ foreground
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        anom_in_fg, connectivity=8
    )

    if num_labels <= 1:
        return 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_cc_area = int(areas.max())
    largest_cc_fg_ratio = largest_cc_area / fg_area

    # Rule 2: anomaly is too fragmented -> treat as noise
    if (largest_cc_fg_ratio < min_largest_cc_fg_ratio) or (
        largest_cc_area < min_largest_cc_pixels
    ):
        return 0

    return 1
