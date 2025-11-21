
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu


def compute_auroc(labels, scores):

    if len(np.unique(labels)) < 2:
        return float('nan')

    return roc_auc_score(labels, scores)


def compute_pro(masks_gt, anomaly_maps, num_thresholds=200):
    if len(masks_gt) == 0 or len(anomaly_maps) == 0:
        return float('nan')

    # Collect all anomaly scores for threshold determination
    all_scores = []
    for anomaly_map in anomaly_maps:
        all_scores.extend(anomaly_map.flatten())

    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    thresholds = np.linspace(min_score, max_score, num_thresholds)

    pros = []

    for threshold in thresholds:
        pro_scores = []

        for mask_gt, anomaly_map in zip(masks_gt, anomaly_maps):
            # Skip images without anomalies
            if mask_gt.sum() == 0:
                continue

            # Resize anomaly map to match ground truth mask
            if anomaly_map.shape != mask_gt.shape:
                anomaly_map_resized = cv2.resize(
                    anomaly_map,
                    (mask_gt.shape[1], mask_gt.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                anomaly_map_resized = anomaly_map

            # Threshold to create binary prediction
            pred_mask = (anomaly_map_resized >= threshold).astype(np.uint8)

            # Find connected components in ground truth
            num_labels, labels_gt = cv2.connectedComponents(mask_gt.astype(np.uint8))

            # Compute overlap for each ground truth region
            for region_id in range(1, num_labels):
                region_mask = (labels_gt == region_id)
                region_size = region_mask.sum()

                if region_size == 0:
                    continue

                # Compute overlap ratio
                overlap = (pred_mask & region_mask).sum() / region_size
                pro_scores.append(overlap)

        if len(pro_scores) > 0:
            pros.append(np.mean(pro_scores))
        else:
            pros.append(0.0)

    # Compute area under PRO curve using trapezoidal rule
    # Limit integration to FPR <= 0.3 (standard practice)
    fpr_limit = 0.3
    fpr_step = 1.0 / num_thresholds
    num_steps_limit = int(fpr_limit / fpr_step)

    # Reverse to get descending thresholds (like FPR)
    pros_limited = pros[::-1][:num_steps_limit]

    if len(pros_limited) > 1:
        aupro = np.trapz(pros_limited, dx=fpr_step) / fpr_limit
    else:
        aupro = 0.0

    return aupro


def compute_pixel_level_metrics(masks_gt, anomaly_maps):
    """Compute pixel-level AUROC and optimal F1 score.

    Use this to evaluate pixel-wise classification performance.

    Args:
        masks_gt: List of ground truth masks (H, W)
        anomaly_maps: List of predicted anomaly maps (H, W)

    Returns:
        dict: {
            'pixel_auroc': AUROC at pixel level,
            'pixel_f1': Maximum F1 score across all thresholds,
            'optimal_threshold': Threshold that achieves max F1
        }
    """
    if len(masks_gt) == 0 or len(anomaly_maps) == 0:
        return {
            'pixel_auroc': np.nan,
            'pixel_f1': np.nan,
            'optimal_threshold': np.nan
        }

    # Flatten all masks and maps
    all_gt = []
    all_pred = []

    for mask_gt, anomaly_map in zip(masks_gt, anomaly_maps):
        # Resize anomaly map if needed
        if anomaly_map.shape != mask_gt.shape:
            anomaly_map_resized = cv2.resize(
                anomaly_map,
                (mask_gt.shape[1], mask_gt.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            anomaly_map_resized = anomaly_map

        all_gt.append(mask_gt.flatten())
        all_pred.append(anomaly_map_resized.flatten())

    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    # Binarize ground truth
    all_gt_binary = (all_gt > 0).astype(int)

    # Compute metrics
    if len(np.unique(all_gt_binary)) < 2:
        return {
            'pixel_auroc': np.nan,
            'pixel_f1': np.nan,
            'optimal_threshold': np.nan
        }

    pixel_auroc = roc_auc_score(all_gt_binary, all_pred)

    # Compute optimal F1 score
    precisions, recalls, thresholds = precision_recall_curve(all_gt_binary, all_pred)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores[:-1])
    pixel_f1 = f1_scores[optimal_idx]
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.0

    return {
        'pixel_auroc': pixel_auroc,
        'pixel_f1': pixel_f1,
        'optimal_threshold': optimal_threshold
    }


def compute_iou(masks_gt, anomaly_maps, method='otsu', smooth_sigma=4):
    if len(masks_gt) == 0 or len(anomaly_maps) == 0:
        return {'mean_iou': np.nan, 'per_image_iou': [], 'threshold': np.nan}

    # Determine optimal threshold if using optimal method
    if method == 'optimal':
        all_gt = []
        all_pred = []

        for mask_gt, anomaly_map in zip(masks_gt, anomaly_maps):
            if anomaly_map.shape != mask_gt.shape:
                anomaly_map_resized = cv2.resize(
                    anomaly_map,
                    (mask_gt.shape[1], mask_gt.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                anomaly_map_resized = anomaly_map

            all_gt.append(mask_gt.flatten())
            all_pred.append(anomaly_map_resized.flatten())

        all_gt = np.concatenate(all_gt)
        all_pred = np.concatenate(all_pred)
        all_gt_binary = (all_gt > 0).astype(int)

        # Find threshold that maximizes F1 (proxy for IoU)
        precisions, recalls, thresholds = precision_recall_curve(all_gt_binary, all_pred)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])
        threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.0
    else:
        threshold = None

    # Compute IoU for each image
    per_image_iou = []

    for mask_gt, anomaly_map in zip(masks_gt, anomaly_maps):
        # Resize anomaly map if needed
        if anomaly_map.shape != mask_gt.shape:
            anomaly_map_resized = cv2.resize(
                anomaly_map,
                (mask_gt.shape[1], mask_gt.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            anomaly_map_resized = anomaly_map

        # Apply Gaussian smoothing
        if smooth_sigma > 0:
            anomaly_map_smooth = gaussian_filter(anomaly_map_resized, sigma=smooth_sigma)
        else:
            anomaly_map_smooth = anomaly_map_resized

        # Threshold to create binary prediction
        if method == 'otsu':
            try:
                thresh = threshold_otsu(anomaly_map_smooth)
            except:
                thresh = np.mean(anomaly_map_smooth)
        else:
            thresh = threshold

        pred_mask = (anomaly_map_smooth >= thresh).astype(np.uint8)
        gt_mask = (mask_gt > 0).astype(np.uint8)

        # Compute IoU
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        per_image_iou.append(iou)

    mean_iou = np.mean(per_image_iou) if len(per_image_iou) > 0 else np.nan

    return {
        'mean_iou': mean_iou,
        'per_image_iou': per_image_iou,
        'threshold': thresh if method == 'otsu' else threshold
    }
