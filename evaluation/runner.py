import numpy as np
from pathlib import Path
from typing import List, Callable, Dict, Any
from tqdm.auto import tqdm

from few_shot.build_memory_bank import build_memory_bank
from few_shot.detect_anomalies import detect_anomalies
from .metrics import compute_auroc, compute_pro, compute_pixel_level_metrics, compute_iou
from .results import MethodResults, ExperimentResults


def run_few_shot(
    model,
    reference_images: List[np.ndarray],
    test_images: List[np.ndarray],
    test_labels: np.ndarray,
    test_masks: List[np.ndarray],
    object_name: str,
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

    for test_img in tqdm(test_images, desc=f"Detecting anomalies ({object_name})"):
        score, anomaly_map, patch_scores = detect_anomalies(
            model=model,
            nn_index=nn_index,
            test_image=test_img,
            use_masking=use_masking,
            k_neighbors=k_neighbors,
            normalize_features=True
        )
        image_scores.append(score)
        anomaly_maps.append(anomaly_map)

    image_scores = np.array(image_scores)

    # Compute image-level metrics
    auroc = compute_auroc(test_labels, image_scores)

    # Compute pixel-level metrics (only for anomalous images)
    anomalous_idx = [i for i, label in enumerate(test_labels) if label == 1]

    if len(anomalous_idx) > 0:
        masks_gt = [test_masks[i] for i in anomalous_idx]
        maps_pred = [anomaly_maps[i] for i in anomalous_idx]

        aupro = compute_pro(masks_gt, maps_pred)
        pixel_metrics = compute_pixel_level_metrics(masks_gt, maps_pred)
        pixel_auroc = pixel_metrics['pixel_auroc']
        pixel_f1 = pixel_metrics['pixel_f1']

        iou_metrics = compute_iou(masks_gt, maps_pred, method='otsu', smooth_sigma=4)
        mean_iou = iou_metrics['mean_iou']
    else:
        aupro = np.nan
        pixel_auroc = np.nan
        pixel_f1 = np.nan
        mean_iou = np.nan

    print(f"  AUROC: {auroc:.4f}, AU-PRO: {aupro:.4f}, IoU: {mean_iou:.4f}")

    return MethodResults(
        method_name=method_name,
        object_name=object_name,
        scores=image_scores,
        maps=anomaly_maps,
        metrics={
            'auroc': auroc,
            'aupro': aupro,
            'pixel_auroc': pixel_auroc,
            'pixel_f1': pixel_f1,
            'mean_iou': mean_iou
        },
        config={
            'k_shot': k_shot,
            'use_rotation': use_rotation,
            'use_masking': use_masking,
            'k_neighbors': k_neighbors
        },
        num_features=num_features
    )


def run_zero_shot(
    model,
    reference_images: List[np.ndarray],
    test_images: List[np.ndarray],
    test_labels: np.ndarray,
    test_masks: List[np.ndarray],
    object_name: str,
    **kwargs
) -> MethodResults:
    raise NotImplementedError(
        "Zero-shot evaluation is not yet implemented. "
        "This is a placeholder for future development. "
        "Use run_few_shot() with k >= 1 for now."
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

    from mvtechDataset import load_mvtec_data

    if method_kwargs is None:
        method_kwargs = {}

    if k_values is None:
        k_values = [8]  # Default to 8-shot

    all_results = {}

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

        # Run method with different k values
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
    from mvtechDataset import load_mvtec_data

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
