import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve


def compute_auroc(labels, scores):

    if len(np.unique(labels)) < 2:
        return float('nan')

    return roc_auc_score(labels, scores)


def compute_pro(masks_gt, anomaly_maps, num_thresholds: int = 200, max_fpr: float = 0.3):

    if len(masks_gt) == 0 or len(anomaly_maps) == 0:
        return float('nan')

    resized_maps = []
    all_scores = []

    for mask_gt, anomaly_map in zip(masks_gt, anomaly_maps):
        if anomaly_map.shape != mask_gt.shape:
            anomaly_map_resized = cv2.resize(
                anomaly_map.astype(np.float32),
                (mask_gt.shape[1], mask_gt.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            anomaly_map_resized = anomaly_map.astype(np.float32)

        resized_maps.append(anomaly_map_resized)
        all_scores.append(anomaly_map_resized.flatten())

    all_scores = np.concatenate(all_scores)

    min_score = float(np.min(all_scores))
    max_score = float(np.max(all_scores))

    if max_score <= min_score:
        # Degenerate case: all scores identical
        return 0.0

    thresholds = np.linspace(min_score, max_score, num_thresholds)

    # Pre-compute total number of negative pixels (GT == 0) in anomaly images
    total_neg_pixels = 0
    for mask_gt in masks_gt:
        if mask_gt.sum() == 0:
            continue
        neg_mask = (mask_gt == 0)
        total_neg_pixels += int(neg_mask.sum())

    if total_neg_pixels == 0:
        return 0.0

    # Scan thresholds: compute PRO and FPR for each
    fpr_list = []
    pro_list = []

    for thr in thresholds:
        region_overlaps = []
        fp_count = 0

        for mask_gt, anomaly_map_resized in zip(masks_gt, resized_maps):
            if mask_gt.sum() == 0:
                continue

            pred_mask = (anomaly_map_resized >= thr)

            num_labels, labels_gt = cv2.connectedComponents(
                (mask_gt > 0).astype(np.uint8)
            )

            for region_id in range(1, num_labels):  # skip background 0
                region_mask = (labels_gt == region_id)
                region_size = int(region_mask.sum())
                if region_size == 0:
                    continue

                tp = np.logical_and(pred_mask, region_mask).sum()
                overlap = tp / float(region_size)
                region_overlaps.append(overlap)

            neg_mask = (mask_gt == 0)
            fp_count += np.logical_and(pred_mask, neg_mask).sum()

        if len(region_overlaps) > 0:
            pro_t = float(np.mean(region_overlaps))
        else:
            pro_t = 0.0

        fpr_t = fp_count / float(total_neg_pixels) if total_neg_pixels > 0 else 0.0

        pro_list.append(pro_t)
        fpr_list.append(fpr_t)

    fpr = np.array(fpr_list)
    pro = np.array(pro_list)

    # Sort by FPR and clip to max_fpr 
    order = np.argsort(fpr)
    fpr = fpr[order]
    pro = pro[order]

    valid = fpr <= max_fpr
    if not np.any(valid):
        return 0.0

    fpr = fpr[valid]
    pro = pro[valid]

    if fpr[0] > 0.0:
        fpr = np.concatenate([[0.0], fpr])
        pro = np.concatenate([[pro[0]], pro])

    aupro = np.trapezoid(pro, fpr) / max_fpr
    aupro = float(np.clip(aupro, 0.0, 1.0))
    return aupro


def compute_pixel_level_metrics(masks_gt, anomaly_maps):
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


def compute_iou(
    masks_gt,
    pred_mask
):
    if len(masks_gt) == 0 or len(pred_mask) == 0:
        return {
            'mean_iou': np.nan,
            'per_image_iou': [],
            'threshold': np.nan,
        }


    # Per-image IoU computation
    per_image_iou = []

    for mask_gt, anomaly_map in zip(masks_gt, pred_mask):
        # Resize anomaly map if needed
        if anomaly_map.shape != mask_gt.shape:
            anomaly_map_resized = cv2.resize(
                anomaly_map.astype(np.float32),
                (mask_gt.shape[1], mask_gt.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            anomaly_map_resized = anomaly_map.astype(np.float32)

        #pred_mask = (pred_mask > 0).astype(np.uint8)
        #masks_gt = (mask_gt > 0).astype(np.uint8)

        # IoU computation 
        intersection = np.logical_and(pred_mask, masks_gt).sum()
        union = np.logical_or(pred_mask, masks_gt).sum()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        per_image_iou.append(iou)

    mean_iou = np.mean(per_image_iou) if len(per_image_iou) > 0 else np.nan

    return {
        'mean_iou': mean_iou,
        'per_image_iou': per_image_iou,
    }
