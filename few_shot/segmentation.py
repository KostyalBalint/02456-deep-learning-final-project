
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve, f1_score


def anomaly_map_to_binary_mask(anomaly_map, threshold=None, method='otsu',
                                smooth_sigma=4, min_area=None, resize_to=None):

    # Apply Gaussian smoothing
    if smooth_sigma > 0:
        anomaly_map_smooth = gaussian_filter(anomaly_map, sigma=smooth_sigma)
    else:
        anomaly_map_smooth = anomaly_map.copy()

    # Normalize to 0-255 range for thresholding
    map_normalized = ((anomaly_map_smooth - anomaly_map_smooth.min()) /
                     (anomaly_map_smooth.max() - anomaly_map_smooth.min() + 1e-8) * 255).astype(np.uint8)

    # Compute threshold
    if threshold is None:
        if method == 'otsu':
            # Otsu's method for automatic thresholding
            threshold, binary_mask = cv2.threshold(
                map_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == 'percentile':
            # Use 95th percentile as threshold
            threshold = np.percentile(anomaly_map_smooth, 95)
            binary_mask = (anomaly_map_smooth > threshold).astype(np.uint8) * 255
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        # Use fixed threshold
        binary_mask = (anomaly_map_smooth > threshold).astype(np.uint8) * 255

    # Remove small connected components
    if min_area is not None and min_area > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        # Keep only components larger than min_area
        for i in range(1, num_labels):  # Skip background (0)
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                binary_mask[labels == i] = 0

    # Resize to match original image size
    if resize_to is not None:
        h, w = resize_to
        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return binary_mask


def find_optimal_threshold(anomaly_maps, ground_truth_masks, metric='f1'):
    # Flatten all maps and masks
    all_scores = []
    all_labels = []

    for amap, gt in zip(anomaly_maps, ground_truth_masks):
        # Resize anomaly map to match ground truth if needed
        if amap.shape != gt.shape:
            amap_resized = cv2.resize(amap, (gt.shape[1], gt.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
        else:
            amap_resized = amap

        # Apply smoothing
        amap_smooth = gaussian_filter(amap_resized, sigma=4)

        all_scores.append(amap_smooth.flatten())
        all_labels.append((gt > 0).astype(int).flatten())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    # Skip if no anomalies present
    if all_labels.sum() == 0:
        return 0.0, 0.0

    # Compute precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_scores)

    if metric == 'f1':
        # Compute F1 scores
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # Exclude last element
        optimal_threshold = thresholds[best_idx]
        best_metric_value = f1_scores[best_idx]
    elif metric == 'iou':
        # Compute IoU (Jaccard index)
        iou_scores = (precisions * recalls) / (precisions + recalls - precisions * recalls + 1e-8)
        best_idx = np.argmax(iou_scores[:-1])
        optimal_threshold = thresholds[best_idx]
        best_metric_value = iou_scores[best_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return optimal_threshold, best_metric_value


def post_process_mask(binary_mask, morph_kernel_size=5, morph_iterations=2):

    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

    # Close small holes
    mask_closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel,
                                   iterations=morph_iterations)

    # Remove small noise
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel,
                                   iterations=1)

    return mask_opened