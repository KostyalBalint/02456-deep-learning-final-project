import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from few_shot.segmentation import anomaly_map_to_binary_mask

def visualize_results(test_image, anomaly_map, ground_truth_mask=None,
                      image_score=None, label=None, sigma=4, vmin=None, vmax=None,
                      show_prediction_mask=True, segmentation_method='otsu'):
    """
    Visualize anomaly detection results.

    Args:
        test_image: Original test image (numpy array H, W, C)
        anomaly_map: Predicted anomaly map (grid_h, grid_w)
        ground_truth_mask: Optional ground truth mask (H, W)
        image_score: Optional image-level anomaly score
        label: Optional ground truth label (0=normal, 1=anomalous)
        sigma: Gaussian blur sigma for smoothing anomaly map
        vmin: Minimum value for colorbar scaling (None = auto)
        vmax: Maximum value for colorbar scaling (None = auto)
        show_prediction_mask: Whether to show predicted binary segmentation mask
        segmentation_method: Method for binary segmentation ('otsu', 'percentile')
    """
    # Apply Gaussian smoothing to anomaly map
    anomaly_map_smooth = gaussian_filter(anomaly_map, sigma=sigma)

    # Resize anomaly map to match image size
    h, w = test_image.shape[:2]
    anomaly_map_resized = cv2.resize(anomaly_map_smooth, (w, h),
                                     interpolation=cv2.INTER_LINEAR)

    # Generate predicted segmentation mask
    if show_prediction_mask:
        predicted_mask = anomaly_map_to_binary_mask(
            anomaly_map,
            method=segmentation_method,
            smooth_sigma=sigma,
            min_area=50,
            resize_to=(h, w)
        )

    # Setup plot
    if ground_truth_mask is None:
        num_plots = 3 if show_prediction_mask else 2
    else:
        num_plots = 4 if show_prediction_mask else 3

    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot original image
    axes[plot_idx].imshow(test_image)
    axes[plot_idx].axis('off')
    title = 'Original Image'
    if label is not None:
        title += f' ({"Anomalous" if label == 1 else "Normal"})'
    axes[plot_idx].set_title(title, fontsize=14)
    plot_idx += 1

    # Plot anomaly heatmap with fixed scale
    im = axes[plot_idx].imshow(anomaly_map_resized, interpolation='bilinear',
                               vmin=vmin, vmax=vmax)
    axes[plot_idx].axis('off')
    title = 'Anomaly Heatmap'
    if image_score is not None:
        title += f'\nScore: {image_score:.4f}'
    axes[plot_idx].set_title(title, fontsize=14)
    plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
    plot_idx += 1

    # Plot predicted segmentation mask with image background
    if show_prediction_mask:
        axes[plot_idx].imshow(test_image)
        axes[plot_idx].imshow(predicted_mask, cmap='gray', alpha=0.5)
        axes[plot_idx].axis('off')
        axes[plot_idx].set_title(f'Predicted Mask ({segmentation_method})', fontsize=14)
        plot_idx += 1

    # Plot ground truth mask with image background
    if ground_truth_mask is not None:
        axes[plot_idx].imshow(test_image)
        axes[plot_idx].imshow(ground_truth_mask, cmap='gray', alpha=0.5)
        axes[plot_idx].axis('off')
        axes[plot_idx].set_title('Ground Truth Mask', fontsize=14)

    plt.tight_layout()
    plt.show()


def visualize_multiple_results(images, anomaly_maps, scores, labels,
                               ground_truth_masks=None, num_display=6, sigma=4,
                               show_prediction_mask=True, segmentation_method='otsu'):
    """
    Visualize results for multiple test images in a grid.

    Args:
        images: List of test images
        anomaly_maps: List of anomaly maps
        scores: List of image-level scores
        labels: List of ground truth labels
        ground_truth_masks: Optional list of ground truth masks
        num_display: Number of images to display
        sigma: Gaussian blur sigma
        show_prediction_mask: Whether to show predicted binary segmentation masks
        segmentation_method: Method for binary segmentation ('otsu', 'percentile')
    """
    import numpy as np

    num_display = min(num_display, len(images))

    # Select diverse examples (normal + anomalous)
    normal_indices = [i for i, l in enumerate(labels) if l == 0]
    anomalous_indices = [i for i, l in enumerate(labels) if l == 1]

    # Select half normal, half anomalous
    num_normal = min(len(normal_indices), num_display // 2)
    num_anomalous = min(len(anomalous_indices), num_display - num_normal)

    selected_indices = (normal_indices[:num_normal] +
                       anomalous_indices[:num_anomalous])

    # Sort selected indices by anomaly score (lowest to highest)
    selected_indices = sorted(selected_indices, key=lambda idx: scores[idx])

    # Compute global min/max across all selected anomaly maps
    selected_maps = [anomaly_maps[idx] for idx in selected_indices]
    vmin = min(np.min(amap) for amap in selected_maps)
    vmax = max(np.max(amap) for amap in selected_maps)

    for idx in selected_indices:
        gt_mask = ground_truth_masks[idx] if ground_truth_masks is not None else None
        visualize_results(images[idx], anomaly_maps[idx], gt_mask,
                         scores[idx], labels[idx], sigma=sigma,
                         vmin=vmin, vmax=vmax,
                         show_prediction_mask=show_prediction_mask,
                         segmentation_method=segmentation_method)