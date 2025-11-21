
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import List, Optional, Dict
from few_shot.segmentation import anomaly_map_to_binary_mask
from .results import MethodResults, ExperimentResults


def plot_single_result(
    test_image: np.ndarray,
    anomaly_map: np.ndarray,
    ground_truth_mask: Optional[np.ndarray] = None,
    image_score: Optional[float] = None,
    label: Optional[int] = None,
    sigma: int = 4,
    show_prediction_mask: bool = True,
    figsize: tuple = (15, 5)
):

    # Apply Gaussian smoothing and resize
    anomaly_map_smooth = gaussian_filter(anomaly_map, sigma=sigma)
    h, w = test_image.shape[:2]
    anomaly_map_resized = cv2.resize(
        anomaly_map_smooth, (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    # Generate predicted mask if requested
    if show_prediction_mask:
        predicted_mask = anomaly_map_to_binary_mask(
            anomaly_map,
            method='otsu',
            smooth_sigma=sigma,
            min_area=50,
            resize_to=(h, w)
        )

    # Setup plot
    num_plots = 2 + int(ground_truth_mask is not None) + int(show_prediction_mask)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Original image
    axes[plot_idx].imshow(test_image)
    axes[plot_idx].axis('off')
    title = 'Original Image'
    if label is not None:
        title += f'\n({"Anomalous" if label == 1 else "Normal"})'
    axes[plot_idx].set_title(title, fontsize=12, fontweight='bold')
    plot_idx += 1

    # Anomaly heatmap
    im = axes[plot_idx].imshow(anomaly_map_resized, interpolation='bilinear', cmap='jet')
    axes[plot_idx].axis('off')
    title = 'Anomaly Heatmap'
    if image_score is not None:
        title += f'\nScore: {image_score:.4f}'
    axes[plot_idx].set_title(title, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
    plot_idx += 1

    # Predicted mask
    if show_prediction_mask:
        axes[plot_idx].imshow(test_image)
        axes[plot_idx].imshow(predicted_mask, cmap='Reds', alpha=0.5)
        axes[plot_idx].axis('off')
        axes[plot_idx].set_title('Predicted Mask\n(Otsu)', fontsize=12, fontweight='bold')
        plot_idx += 1

    # Ground truth mask
    if ground_truth_mask is not None:
        axes[plot_idx].imshow(test_image)
        axes[plot_idx].imshow(ground_truth_mask, cmap='Reds', alpha=0.5)
        axes[plot_idx].axis('off')
        axes[plot_idx].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_result_grid(
    results: MethodResults,
    test_images: List[np.ndarray],
    test_labels: np.ndarray,
    test_masks: List[np.ndarray],
    num_display: int = 6,
    sigma: int = 4
):

    # Select diverse examples (normal + anomalous, sorted by score)
    normal_indices = [i for i, l in enumerate(test_labels) if l == 0]
    anomalous_indices = [i for i, l in enumerate(test_labels) if l == 1]

    num_normal = min(len(normal_indices), num_display // 2)
    num_anomalous = min(len(anomalous_indices), num_display - num_normal)

    selected_indices = (
        normal_indices[:num_normal] +
        anomalous_indices[:num_anomalous]
    )
    selected_indices = sorted(selected_indices, key=lambda idx: results.scores[idx])

    # Compute global min/max for consistent colormap
    selected_maps = [results.maps[idx] for idx in selected_indices]
    vmin = min(np.min(m) for m in selected_maps)
    vmax = max(np.max(m) for m in selected_maps)

    for idx in selected_indices:
        plot_single_result(
            test_image=test_images[idx],
            anomaly_map=results.maps[idx],
            ground_truth_mask=test_masks[idx],
            image_score=results.scores[idx],
            label=test_labels[idx],
            sigma=sigma,
            show_prediction_mask=True
        )



def plot_method_comparison(
    results: ExperimentResults,
    metrics: List[str] = ['auroc', 'aupro'],
    figsize: tuple = (15, 5)
):

    summary_df = results.summary_table()

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    if num_metrics == 1:
        axes = [axes]

    metric_labels = {
        'auroc': ('Mean AUROC', 'Std AUROC', 'Image-level Classification'),
        'aupro': ('Mean AU-PRO', 'Std AU-PRO', 'Pixel-level Localization'),
        'pixel_auroc': ('Mean Pixel AUROC', None, 'Pixel-level AUROC'),
        'pixel_f1': ('Mean Pixel F1', None, 'Pixel-level F1'),
        'mean_iou': ('Mean IoU', None, 'Segmentation IoU')
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        mean_col, std_col, title = metric_labels.get(metric, (metric, None, metric))

        if mean_col not in summary_df.columns:
            ax.text(0.5, 0.5, f'{metric} not available', ha='center', va='center')
            ax.set_title(title)
            continue

        methods = summary_df['Method'].values
        means = summary_df[mean_col].values
        stds = summary_df[std_col].values if std_col and std_col in summary_df.columns else None

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)

        ax.set_xlabel('Method', fontsize=10)
        ax.set_ylabel(mean_col, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_object_heatmap(
    results: ExperimentResults,
    metrics: List[str] = ['AUROC', 'AU-PRO'],
    figsize: tuple = (14, 10)
):

    per_object_df = results.per_object_table()

    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=figsize)
    if num_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        if metric not in per_object_df.columns:
            ax.text(0.5, 0.5, f'{metric} not available', ha='center', va='center')
            ax.set_title(metric)
            continue

        pivot = per_object_df.pivot_table(
            index='Object',
            columns='Method',
            values=metric
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={'label': metric}
        )
        ax.set_title(f'{metric} by Object and Method', fontsize=12, fontweight='bold')
        ax.set_xlabel('')

    plt.tight_layout()
    plt.show()


def plot_difference_heatmap(
    results1: ExperimentResults,
    results2: ExperimentResults,
    method1: str,
    method2: str,
    metric: str = 'AUROC',
    figsize: tuple = (10, 8)
):
    df1 = results1.per_object_table()
    df2 = results2.per_object_table()

    # Filter to selected methods
    df1 = df1[df1['Method'] == method1].set_index('Object')
    df2 = df2[df2['Method'] == method2].set_index('Object')

    # Compute difference
    diff = df2[metric] - df1[metric]

    fig, ax = plt.subplots(figsize=figsize)
    diff.plot(kind='barh', ax=ax, color=diff.apply(lambda x: 'green' if x > 0 else 'red'))
    ax.set_xlabel(f'Difference in {metric} ({method2} - {method1})')
    ax.set_title(f'Performance Difference: {method2} vs {method1}', fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_shot_count_analysis(
    results: ExperimentResults,
    k_values: List[int],
    metrics: List[str] = ['auroc', 'aupro'],
    figsize: tuple = (15, 5)
):

    summary_df = results.summary_table()

    # Filter to k-shot methods
    k_shot_methods = [f"{k}_shot" for k in k_values]
    k_shot_data = summary_df[summary_df['Method'].isin(k_shot_methods)].copy()
    k_shot_data['k'] = k_shot_data['Method'].str.extract(r'(\d+)_shot').astype(int)
    k_shot_data = k_shot_data.sort_values('k')

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    if num_metrics == 1:
        axes = [axes]

    metric_info = {
        'auroc': ('Mean AUROC', 'Std AUROC', 'Classification vs Shot Count'),
        'aupro': ('Mean AU-PRO', 'Std AU-PRO', 'Localization vs Shot Count'),
        'mean_iou': ('Mean IoU', None, 'Segmentation vs Shot Count')
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        mean_col, std_col, title = metric_info.get(metric, (metric, None, metric))

        if mean_col not in k_shot_data.columns:
            continue

        k_values_actual = k_shot_data['k'].values
        means = k_shot_data[mean_col].values
        stds = k_shot_data[std_col].values if std_col and std_col in k_shot_data.columns else None

        ax.plot(k_values_actual, means, 'o-', linewidth=2, markersize=8)

        if stds is not None:
            ax.fill_between(
                k_values_actual,
                means - stds,
                means + stds,
                alpha=0.3
            )

        ax.set_xlabel('Number of Reference Images (k)', fontsize=10)
        ax.set_ylabel(mean_col, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.show()

    # Print improvement statistics
    if len(k_shot_data) > 1:
        first_k = k_shot_data.iloc[0]
        best_k = k_shot_data.iloc[-1]

        print(f"\nImprovement from {int(first_k['k'])}-shot to {int(best_k['k'])}-shot:")
        for metric in metrics:
            mean_col = metric_info.get(metric, (metric, None, metric))[0]
            if mean_col in k_shot_data.columns:
                improvement = ((best_k[mean_col] - first_k[mean_col]) / first_k[mean_col]) * 100
                print(f"  {metric.upper()}: {first_k[mean_col]:.4f} → {best_k[mean_col]:.4f} (+{improvement:.1f}%)")



def save_all_plots(
    results: ExperimentResults,
    output_dir: Path,
    k_values: List[int] = None
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Method comparison
    plt.figure()
    plot_method_comparison(results)
    plt.savefig(output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Object heatmap
    plt.figure()
    plot_object_heatmap(results)
    plt.savefig(output_dir / 'object_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Shot count analysis (if applicable)
    if k_values and len(k_values) > 1:
        plt.figure()
        plot_shot_count_analysis(results, k_values)
        plt.savefig(output_dir / 'shot_count_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"All plots saved to {output_dir}")
