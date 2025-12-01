
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve, f1_score
import hdbscan
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture

def anomaly_map_to_binary_mask(
    anomaly_map,
    method,  # 'percentile' or 'gmm'
    threshold=None,
    smooth_sigma=4,
    resize_to=None,
    percentile=95.0,
    gmm_min_points=100,
):
    roi_mask = anomaly_map != 0

    # No foreground at all: return empty mask
    if not np.any(roi_mask):
        binary_mask = np.zeros_like(anomaly_map, dtype=np.uint8)
        if resize_to is not None and resize_to != anomaly_map.shape:
            h, w = resize_to
            binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return binary_mask

    if smooth_sigma > 0:
        anomaly_map_smooth = gaussian_filter(anomaly_map, sigma=smooth_sigma)
    else:
        anomaly_map_smooth = anomaly_map.copy()

    binary_mask = np.zeros_like(anomaly_map_smooth, dtype=np.uint8)
    roi_vals = anomaly_map_smooth[roi_mask]

    if method == "percentile":
        if threshold is None:
            threshold = np.percentile(roi_vals, percentile)
        else:
            threshold = float(threshold)
            
        binary_mask[roi_mask] = (roi_vals > threshold).astype(np.uint8) * 255

    elif method == "gmm":
        # If too few points, GMM is unstable -> fall back to percentile
        if roi_vals.size < gmm_min_points:
            threshold = np.percentile(roi_vals, percentile)
            binary_mask[roi_mask] = (roi_vals > threshold).astype(np.uint8) * 255
        else:
            vals = roi_vals.reshape(-1, 1).astype(np.float32)

            gmm = GaussianMixture(
                n_components=2,
                covariance_type="full",
                reg_covar=1e-6,
                random_state=0,
            )
            gmm.fit(vals)

            means = gmm.means_.flatten()
            # component with higher mean is treated as "anomaly"
            anomaly_comp = int(np.argmax(means))

            comp_labels = gmm.predict(vals)  # 0 or 1 for each ROI pixel
            anomaly_mask_flat = (comp_labels == anomaly_comp)

            binary_mask[roi_mask] = anomaly_mask_flat.astype(np.uint8) * 255

    else:
        raise ValueError(f"Unknown method: {method}")

    binary_mask[~roi_mask] = 0

    """    # --- 3) Remove small connected components (optional) ---
    if min_area is not None and min_area > 0:
        num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=16
        )
        for i in range(1, num_labels):  # skip background 0
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                binary_mask[labels_cc == i] = 0"""

    
    # --- 4) Optional resize ---
    if resize_to is not None and resize_to != anomaly_map.shape:
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