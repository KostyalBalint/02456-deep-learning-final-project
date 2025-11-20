import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

from few_shot.DINOv3Wrapper import DINOv3Wrapper
from few_shot.build_memory_bank import build_memory_bank
from few_shot.detect_anomalies import detect_anomalies
from few_shot.segmentation import anomaly_map_to_binary_mask
from mvtechDataset import ensure_mvtech_dataset_is_downloaded, get_categories, load_mvtec_data


def save_visualization_results(images, anomaly_maps, scores, labels,
                                ground_truth_masks=None, num_display=6, sigma=4,
                                show_prediction_mask=True, segmentation_method='otsu',
                                output_path='output.png'):
    """
    Save visualization results for multiple test images in a single figure.

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
        output_path: Path to save the output image
    """
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

    # Determine number of columns
    if ground_truth_masks is None:
        num_cols = 3 if show_prediction_mask else 2
    else:
        num_cols = 4 if show_prediction_mask else 3

    num_rows = len(selected_indices)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    if num_rows == 1:
        axes = [axes]

    for row, idx in enumerate(selected_indices):
        test_image = images[idx]
        anomaly_map = anomaly_maps[idx]
        image_score = scores[idx]
        label = labels[idx]
        gt_mask = ground_truth_masks[idx] if ground_truth_masks is not None else None

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

        col_idx = 0
        ax_row = axes[row] if num_rows > 1 else axes[0]

        # Plot original image
        ax_row[col_idx].imshow(test_image)
        ax_row[col_idx].axis('off')
        title = 'Original Image'
        if label is not None:
            title += f' ({"Anomalous" if label == 1 else "Normal"})'
        ax_row[col_idx].set_title(title, fontsize=12)
        col_idx += 1

        # Plot anomaly heatmap
        im = ax_row[col_idx].imshow(anomaly_map_resized, interpolation='bilinear',
                                    vmin=vmin, vmax=vmax)
        ax_row[col_idx].axis('off')
        title = 'Anomaly Heatmap'
        if image_score is not None:
            title += f'\nScore: {image_score:.4f}'
        ax_row[col_idx].set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax_row[col_idx], fraction=0.046, pad=0.04)
        col_idx += 1

        # Plot predicted segmentation mask
        if show_prediction_mask:
            ax_row[col_idx].imshow(test_image)
            ax_row[col_idx].imshow(predicted_mask, cmap='gray', alpha=0.5)
            ax_row[col_idx].axis('off')
            ax_row[col_idx].set_title(f'Predicted Mask ({segmentation_method})', fontsize=12)
            col_idx += 1

        # Plot ground truth mask
        if gt_mask is not None:
            ax_row[col_idx].imshow(test_image)
            ax_row[col_idx].imshow(gt_mask, cmap='gray', alpha=0.5)
            ax_row[col_idx].axis('off')
            ax_row[col_idx].set_title('Ground Truth Mask', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


def process_category(model, dataset_path, category, num_shots, output_dir,
                     num_display=6, max_test_images=None):
    """
    Process a single category: build memory bank, detect anomalies, save visualization.

    Args:
        model: DINOv3Wrapper model
        dataset_path: Path to MVTec dataset
        category: Category name
        num_shots: Number of reference images
        output_dir: Directory to save outputs
        num_display: Number of images to display in visualization
        max_test_images: Maximum number of test images to process (None = all)
    """
    print(f"\n{'='*60}")
    print(f"Processing category: {category}")
    print(f"{'='*60}")

    # Load data
    reference_images, test_images, test_labels, test_masks = load_mvtec_data(
        dataset_path, category, num_shots
    )

    # Build memory bank
    nn_index, features_ref, num_features = build_memory_bank(
        model=model,
        reference_images=reference_images,
        use_rotation=True,
        normalize_features=True
    )
    print(f"Memory bank ready with {num_features} patches!")

    # Run anomaly detection
    if max_test_images is not None:
        test_images = test_images[:max_test_images]
        test_labels = test_labels[:max_test_images]
        test_masks = test_masks[:max_test_images]

    print(f"Running anomaly detection on {len(test_images)} test images...")

    image_scores = []
    anomaly_maps = []

    for test_img in tqdm(test_images, desc="Detecting anomalies"):
        score, anomaly_map, patch_scores = detect_anomalies(
            model=model,
            nn_index=nn_index,
            test_image=test_img,
            use_masking=True,
            k_neighbors=3,
            normalize_features=True
        )
        image_scores.append(score)
        anomaly_maps.append(anomaly_map)

    image_scores = np.array(image_scores)
    test_labels = np.array(test_labels)

    print("Detection complete!")

    # Save visualization
    output_path = os.path.join(output_dir, f"{category}_results.png")
    save_visualization_results(
        images=test_images,
        anomaly_maps=anomaly_maps,
        scores=image_scores,
        labels=test_labels,
        ground_truth_masks=test_masks,
        num_display=num_display,
        sigma=0,
        segmentation_method='percentile',
        output_path=output_path
    )

    return image_scores, test_labels


def main():
    parser = argparse.ArgumentParser(description='Run few-shot anomaly detection on all MVTec categories')
    parser.add_argument('--output-dir', type=str, default='./output_visualizations',
                        help='Directory to save output visualizations')
    parser.add_argument('--num-shots', type=int, default=4,
                        help='Number of reference images per category')
    parser.add_argument('--num-display', type=int, default=6,
                        help='Number of images to display per category')
    parser.add_argument('--max-test-images', type=int, default=None,
                        help='Maximum test images to process per category (default: all)')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='Specific categories to process (default: all)')
    parser.add_argument('--model-name', type=str,
                        default='facebook/dinov3-vits16-pretrain-lvd1689m',
                        help='DINO model name')
    parser.add_argument('--smaller-edge-size', type=int, default=1024,
                        help='Smaller edge size for image processing')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Download/verify dataset
    dataset_path = ensure_mvtech_dataset_is_downloaded(".")

    # Get categories to process
    all_categories = get_categories(dataset_path)
    if args.categories:
        categories = [c for c in args.categories if c in all_categories]
        if not categories:
            print(f"Error: None of the specified categories found. Available: {all_categories}")
            return
    else:
        categories = all_categories

    print(f"Categories to process: {categories}")

    # Load model
    print("\nLoading DINO model...")
    model = DINOv3Wrapper(
        model_name=args.model_name,
        smaller_edge_size=args.smaller_edge_size
    )

    # Process each category
    results = {}
    for category in categories:
        try:
            scores, labels = process_category(
                model=model,
                dataset_path=dataset_path,
                category=category,
                num_shots=args.num_shots,
                output_dir=args.output_dir,
                num_display=args.num_display,
                max_test_images=args.max_test_images
            )
            results[category] = {'scores': scores, 'labels': labels}
        except Exception as e:
            print(f"Error processing {category}: {e}")
            continue

    print(f"\n{'='*60}")
    print("All categories processed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()