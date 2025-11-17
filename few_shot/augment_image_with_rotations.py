import cv2
import numpy as np
from PIL import Image

def augment_image_with_rotations(image):
    """
    Augment image with 8-way rotations.
    Based on AnomalyDINO's augmentation strategy.

    Args:
        image: PIL Image or numpy array (H, W, C)

    Returns:
        List of 8 augmented images (rotations at 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    """
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()

    augmented = []

    # 8 rotations: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        if angle == 0:
            rotated = image_np
        else:
            # Get rotation matrix
            h, w = image_np.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Rotate with border replication
            rotated = cv2.warpAffine(image_np, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

        augmented.append(rotated)

    return augmented
