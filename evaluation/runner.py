import cv2
import numpy as np
from pathlib import Path
from typing import List, Callable, Dict, Any
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm.auto import tqdm

from utils.build_memory_bank import build_memory_bank
from .metrics import compute_auroc, compute_pro, compute_pixel_level_metrics, compute_iou
from .results import MethodResults, ExperimentResults

from few_shot.detect_anomalies import detect_anomalies as few_shot_detect_anomalies
from zero_shot.detect_anomalies import detect_anomalies as zero_shot_detect_anomalies
from utils.segmentation import anomaly_map_to_binary_mask
from utils.classification import classify_image_from_masks


def run_few_shot(
    model,
    reference_images: List[np.ndarray],
    test_images: List[np.ndarray],
    test_labels: np.ndarray,
    test_masks: List[np.ndarray],
    object_name: str,
    masking_method: str,
    segmentation_method: str,
    use_rotation: bool = False,
    use_masking: bool = False,
    k_neighbors: int = 3
) -> MethodResults:
    k_shot = len(reference_images)
    method_name = f"{k_shot}_shot"

    print(f"Running {method_name} evaluation on {object_name}...")

    # Build memory bank from reference images
    nn_index, features_ref, num_features = build_memory_bank(
        model=model,
        reference_images=reference_images,
        use_rotation=use_rotation,
        normalize_features=True
    )

    # Detect anomalies in test images
    image_scores = []
    anomaly_maps = []
    mask_grids = []

    for test_img in tqdm(test_images, desc=f"Detecting anomalies ({object_name})"):
        score, anomaly_map, patch_scores, mask_grid = few_shot_detect_anomalies(
            model=model,
            nn_index=nn_index,
            test_image=test_img,
            use_masking=use_masking,
            masking_method=masking_method,
            k_neighbors=k_neighbors,
            normalize_features=True
        )
        image_scores.append(score)
        anomaly_maps.append(anomaly_map)
        mask_grids.append(mask_grid)

    return compute_results(
        method_name=method_name,
        object_name=object_name,
        image_scores=image_scores,
        anomaly_maps=anomaly_maps,
        fg_mask_grids=mask_grids,
        segmentation_method=segmentation_method,
        test_labels=test_labels,
        test_masks=test_masks,
        num_features=num_features,
        use_rotation=use_rotation,
        use_masking=use_masking,
        k_neighbors=k_neighbors,
        masking_method=masking_method
    )


def run_zero_shot(
    model,
    reference_images: List[np.ndarray],  # Ignored - for API compatibility with run_evaluation
    test_images: List[np.ndarray],
    test_labels: np.ndarray,
    test_masks: List[np.ndarray],
    object_name: str,
    masking_method: str,
    segmentation_method: str,
    use_masking: bool = False,
    **kwargs
) -> MethodResults:
    
    method_name = "zero_shot"

    print(f"Running {method_name} evaluation on {object_name}...")
    
    # Detect anomalies in test images
    image_scores = []
    anomaly_maps = []
    mask_grids = []
    
    # Reference images are not used in zero-shot
    for idx, test_img in enumerate(tqdm(test_images, desc=f"Detecting anomalies ({object_name})")):
        reference_images = test_images[:idx] + test_images[idx+1:]
        nn_index, features_ref, num_features = build_memory_bank(
            model=model,
            reference_images=reference_images,
            use_rotation=False,
            normalize_features=True
        )
        score, anomaly_map, patch_scores, mask_grid = zero_shot_detect_anomalies(
            model=model,
            nn_index=nn_index,
            test_image=test_img,
            use_masking=use_masking,
            masking_method=masking_method,
            normalize_features=True
        )
        image_scores.append(score)
        anomaly_maps.append(anomaly_map)
        mask_grids.append(mask_grid)

    return compute_results(
        method_name=method_name,
        object_name=object_name,
        image_scores=image_scores,
        anomaly_maps=anomaly_maps,
        fg_mask_grids=mask_grids,
        segmentation_method=segmentation_method,
        test_labels=test_labels,
        test_masks=test_masks,
        num_features=num_features,
        use_masking=use_masking,
        **kwargs
    )

def compute_results(method_name,
                    object_name,
                    image_scores,
                    anomaly_maps,
                    fg_mask_grids,
                    segmentation_method,
                    test_labels,
                    test_masks,
                    num_features,
                    **kwargs
) -> MethodResults: 
    image_scores = np.array(image_scores)
    test_labels = np.array(test_labels)
    num_images = len(anomaly_maps)

    # Compute image-level metrics
    auroc = compute_auroc(test_labels, image_scores)

    pred_anomaly_masks = []
    
    for i in range(num_images):

        # Use GT mask size if available, otherwise anomaly map size
        amap = anomaly_maps[i]
        if test_masks is not None and test_masks[i] is not None:
            h, w = test_masks[i].shape
        else:
            h, w = amap.shape

        bin_mask = anomaly_map_to_binary_mask(
            anomaly_maps[i],
            threshold=None,
            method=segmentation_method,
            smooth_sigma=4,
            resize_to=(h, w),
        )
        pred_anomaly_masks.append(bin_mask)

    # Compute pixel-level metrics (only for anomalous images)
    anomalous_idx = [i for i, label in enumerate(test_labels) if label == 1]

    if len(anomalous_idx) > 0:
        masks_gt = [test_masks[i] for i in anomalous_idx]
        maps_pred = [anomaly_maps[i] for i in anomalous_idx]

        aupro = compute_pro(masks_gt, maps_pred)
        pixel_metrics = compute_pixel_level_metrics(masks_gt, maps_pred)
        pixel_auroc = pixel_metrics['pixel_auroc']
        pixel_f1 = pixel_metrics['pixel_f1']
        
        pred_masks_anom = [
            (pred_anomaly_masks[i] > 0).astype(np.uint8) for i in anomalous_idx
        ]        
        iou_metrics = compute_iou(masks_gt, pred_masks_anom)
        mean_iou = iou_metrics['mean_iou']
    else:
        aupro = np.nan
        pixel_auroc = np.nan
        pixel_f1 = np.nan
        mean_iou = np.nan

    #print(f"  AUROC: {auroc:.4f}, AU-PRO: {aupro:.4f}, IoU: {mean_iou:.4f}")
    
    pred_labels = []


    for i in range(num_images):
        anom_mask = pred_anomaly_masks[i]
        fg_mask = fg_mask_grids[i]
        H, W = anom_mask.shape
        
        fg_mask_resized = cv2.resize(
            (fg_mask > 0).astype(np.uint8),
            (W, H),  # cv2 expects (width, height)
            interpolation=cv2.INTER_NEAREST,
        )

        pred = classify_image_from_masks(
            anomaly_mask=anom_mask,
            foreground_mask=fg_mask_resized
        )
        pred_labels.append(pred)

    pred_labels = np.array(pred_labels, dtype=int)
    
    if len(np.unique(test_labels)) < 2:
        cls_accuracy = np.nan
        cls_precision = np.nan
        cls_recall = np.nan
        cls_f1 = np.nan
        tp = fp = fn = tn = 0
    else:
        cls_accuracy = float((pred_labels == test_labels).mean())

        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels,
            pred_labels,
            average='binary',
            zero_division=0
        )
        cls_precision = float(precision)
        cls_recall = float(recall)
        cls_f1 = float(f1)

        tn, fp, fn, tp = confusion_matrix(
            test_labels, pred_labels, labels=[0, 1]
        ).ravel()

    '''print(
        f"  AUROC: {auroc:.4f}, AU-PRO: {aupro:.4f}, IoU: {mean_iou:.4f}, "
        f"Cls Acc: {cls_accuracy:.4f}, Cls Recall: {cls_recall:.4f}"
    )'''


    config = dict(kwargs)
    config.setdefault('segmentation_method', segmentation_method)

    return MethodResults(
        method_name=method_name,
        object_name=object_name,
        scores=image_scores,
        maps=anomaly_maps,
        metrics={
            "auroc": auroc,
            "aupro": aupro,
            "pixel_auroc": pixel_auroc,
            "pixel_f1": pixel_f1,
            "mean_iou": mean_iou,
            "cls_accuracy": cls_accuracy,
            "cls_precision": cls_precision,
            "cls_recall": cls_recall,
            "cls_f1": cls_f1,
            "cls_tp": int(tp),
            "cls_fp": int(fp),
            "cls_fn": int(fn),
            "cls_tn": int(tn),
        },
        config=config,
        num_features=num_features
    )


def run_evaluation(
    method_fn: Callable,
    method_name: str,
    model,
    dataset_path: Path,
    objects: List[str],
    k_values: List[int] = None,
    method_kwargs: Dict[str, Any] = None
) -> ExperimentResults:

    from utils.mvtechDataset import load_mvtec_data

    if method_kwargs is None:
        method_kwargs = {}

    if k_values is None:
        k_values = [8]  # Default to 8-shot

    all_results = {}
    
    print(f"Evaluation of anomaly detection with:  {method_name}")

    for obj in objects:
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {obj}")
        print(f"{'=' * 70}\n")

        all_results[obj] = {}

        # Load data (use max k for reference images)
        max_k = max(k_values) if k_values else 8
        reference_images, test_images, test_labels, test_masks = load_mvtec_data(
            dataset_path, obj, num_shots=max_k
        )

        for k in k_values:
            try:
                result = method_fn(
                    model=model,
                    reference_images=reference_images[:k],
                    test_images=test_images,
                    test_labels=test_labels,
                    test_masks=test_masks,
                    object_name=obj,
                    **method_kwargs
                )
                all_results[obj][result.method_name] = result

            except Exception as e:
                print(f"  {k}-shot failed: {e}")
                import traceback
                traceback.print_exc()

    return ExperimentResults(results=all_results)


def run_single_object(
    method_fn: Callable,
    model,
    dataset_path: Path,
    object_name: str,
    k: int = 8,
    method_kwargs: Dict[str, Any] = None
) -> MethodResults:
    from utils.mvtechDataset import load_mvtec_data

    if method_kwargs is None:
        method_kwargs = {}

    reference_images, test_images, test_labels, test_masks = load_mvtec_data(
        dataset_path, object_name, num_shots=k
    )

    return method_fn(
        model=model,
        reference_images=reference_images[:k],
        test_images=test_images,
        test_labels=test_labels,
        test_masks=test_masks,
        object_name=object_name,
        **method_kwargs
    )
