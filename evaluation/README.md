# Evaluation Module - Refactored

This module provides a clean, extensible API for evaluating anomaly detection methods on the MVTec dataset.

## Quick Start

```python
from evaluation import run_evaluation, run_few_shot, ExperimentResults
from few_shot.DINOv3Wrapper import DINOv3Wrapper
from pathlib import Path

# Load model
model = DINOv3Wrapper('facebook/dinov3-vits16-pretrain-lvd1689m')

# Run evaluation
results = run_evaluation(
    method_fn=run_few_shot,
    method_name='few_shot',
    model=model,
    dataset_path=Path('./mvtec_anomaly_detection'),
    objects=['bottle', 'hazelnut'],
    k_values=[1, 4, 8],
    method_kwargs={'use_rotation': True}
)

# Analyze
print(results.summary_table())

# Save
results.save(Path('./results/experiment_1'))
```

## Module Structure

```
evaluation/
├── __init__.py          # Public API exports
├── metrics.py           # Metric computations (AUROC, AU-PRO, IoU, etc.)
├── runner.py            # Experiment execution
├── results.py           # Result management (save/load/compare)
├── visualization.py     # Plotting functions
└── export.py            # Export to JSON/CSV/LaTeX/Markdown
```

## Key Classes

### `MethodResults`

Stores results from evaluating a single method on a single object.

```python
@dataclass
class MethodResults:
    method_name: str          # e.g., '8_shot'
    object_name: str          # e.g., 'hazelnut'
    scores: np.ndarray        # Image-level anomaly scores
    maps: List[np.ndarray]    # Anomaly maps
    metrics: Dict[str, float] # Computed metrics
    config: Dict[str, Any]    # Method parameters
```

### `ExperimentResults`

Manages results across multiple objects and methods.

```python
results = ExperimentResults.load(path)

# Query
results.list_methods()                    # ['1_shot', '4_shot', '8_shot']
results.list_objects()                    # ['bottle', 'hazelnut', ...]
results.get_method('8_shot')              # All objects for 8-shot
results.get_object('hazelnut')            # All methods for hazelnut

# Analyze
results.summary_table()                   # Mean/std across objects
results.per_object_table()                # Detailed breakdown
results.compare_methods('1_shot', '8_shot', metric='auroc')

# Save
results.save(Path('./results'))
```

## Adding a New Method

**Step 1:** Implement your method following the standard signature:

```python
def run_my_method(
    model,
    reference_images: List[np.ndarray],
    test_images: List[np.ndarray],
    test_labels: np.ndarray,
    test_masks: List[np.ndarray],
    object_name: str,
    **kwargs
) -> MethodResults:
    # Your implementation here
    scores = ...
    maps = ...

    # Compute standard metrics
    from evaluation import compute_auroc, compute_pro, compute_iou
    auroc = compute_auroc(test_labels, scores)
    # ... compute other metrics ...

    return MethodResults(
        method_name='my_method',
        object_name=object_name,
        scores=scores,
        maps=maps,
        metrics={'auroc': auroc, ...},
        config=kwargs
    )
```

**Step 2:** Run evaluation:

```python
results = run_evaluation(
    method_fn=run_my_method,
    method_name='my_method',
    model=model,
    dataset_path=dataset_path,
    objects=categories,
    k_values=[8],
    method_kwargs={'my_param': value}
)
```

**Step 3:** Compare with baseline:

```python
baseline = ExperimentResults.load('./baseline_results')
comparison = results.compare_methods('my_method', baseline, '8_shot')
print(comparison)
```

## Metrics

### Image-Level (Classification)
- **`compute_auroc(labels, scores)`**: Area Under ROC Curve

### Pixel-Level (Localization)
- **`compute_pro(masks_gt, anomaly_maps)`**: Area Under Per-Region-Overlap curve
- **`compute_pixel_level_metrics(masks_gt, anomaly_maps)`**: Pixel AUROC and F1

### Segmentation
- **`compute_iou(masks_gt, anomaly_maps)`**: Intersection over Union

## Visualization

```python
from evaluation import (
    plot_single_result,
    plot_method_comparison,
    plot_object_heatmap,
    plot_shot_count_analysis
)

# Single image
plot_single_result(image, anomaly_map, gt_mask, score, label)

# Method comparison
plot_method_comparison(results, metrics=['auroc', 'aupro'])

# Per-object heatmap
plot_object_heatmap(results)

# Shot count analysis
plot_shot_count_analysis(results, k_values=[1, 2, 4, 8])
```

## Export

```python
from evaluation import export_all

# Export everything (JSON, CSV, LaTeX, Markdown)
export_all(results, Path('./export'))

# Or individual formats
from evaluation import export_json, export_summary_csv, generate_markdown_report
export_json(results, './results.json')
export_summary_csv(results, './summary.csv')
generate_markdown_report(results, './report.md')
```