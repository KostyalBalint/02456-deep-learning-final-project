from typing import List
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import torch
import argparse
import os
import yaml
from sklearn.metrics import roc_auc_score
import cv2
from datetime import datetime
import json
import time
import faiss

from utils.DINOv3Wrapper import DINOv3Wrapper
from utils.mvtechDataset import ensure_mvtech_dataset_is_downloaded, get_categories, get_subcategories


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument('--config', type=str, required=True)
    
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    with open(args.config, 'r') as f:
        config_args = yaml.safe_load(f)
    for key, value in config_args.items():
        setattr(args, key, value)

    if args.model_size not in ['s16', 'b16', 'l16', '7b16', 's16plus']:
        raise ValueError("model_size must be one of 's', 'b', or 'l'")

    return args


def load_image(image_path: str) -> Image.Image:
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def batched_zero_shot_anomaly_detection(images: List[Image.Image], model: DINOv3Wrapper, use_masking: bool):
    """
    Notebook-style implementation with main.py's feature extraction.
    
    Args:
        images: List of PIL images
        model: DINOv3Wrapper instance
        use_masking: Whether to apply background masking
    
    Returns:
        image_level_scores: Array of anomaly scores per image
        timing_stats: Dictionary with timing information
    """
    timing_stats = {}
    num_images = len(images)
    
    # Extract features for all images (like notebook)
    all_patch_features = []
    all_masks = []
    
    feature_extraction_start = time.time()
    for image in tqdm(images, desc="Extracting features", total=num_images):
        # Use main.py's feature extraction approach
        image_tensor, grid_size = model.prepare_image(image)
        features = model.extract_features(image_tensor)
        mask = model.compute_background_mask(features, grid_size, masking_type=use_masking)
        
        all_patch_features.append(features)
        all_masks.append(mask)
    
    feature_extraction_time = time.time() - feature_extraction_start
    timing_stats['feature_extraction'] = feature_extraction_time
    print(f"  Feature extraction: {feature_extraction_time:.2f}s")
    
    # Convert to 3D numpy array for efficient indexing
    array_conversion_start = time.time()
    all_patch_features = np.array(all_patch_features)
    array_conversion_time = time.time() - array_conversion_start
    timing_stats['array_conversion'] = array_conversion_time
    print(f"  Array conversion: {array_conversion_time:.2f}s")
    print(f'  All patch features shape: {all_patch_features.shape}')
    
    hidden_dim = all_patch_features.shape[-1]
    image_level_scores = []
    
    # Distance calculation loop (using FAISS for efficient similarity search)
    distance_calc_start = time.time()
    for i in tqdm(range(num_images), desc="Calculating distances"):
        # 1. Select the patches for the current test image
        test_patches = all_patch_features[i, :, :]  # Shape: (num_patches, hidden_dim)
        
        # 2. Create the memory bank from ALL OTHER images
        filter_i = np.arange(num_images) != i
        memory_bank_patches = all_patch_features[filter_i]
        
        # Reshape bank to ( (num_images - 1) * num_patches, hidden_dim )
        memory_bank = memory_bank_patches.reshape(-1, hidden_dim).astype('float32')
        
        # 3. Calculate cosine distances using FAISS
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(test_patches)
        faiss.normalize_L2(memory_bank)
        
        # Build FAISS index for the memory bank
        index = faiss.IndexFlatIP(hidden_dim)  # Inner Product (cosine similarity after normalization)
        index.add(memory_bank)
        
        # 4: Calculate Patch-Level Scores
        # For EACH test patch, find the mean of its 0.1% lowest distances to the bank
        k_patch = max(1, int(0.001 * memory_bank.shape[0]))
        
        # Search for k_patch nearest neighbors (highest similarities)
        similarities, indices = index.search(test_patches, k_patch)
        
        # Convert similarities to distances (1 - similarity)
        lowest_k_distances = 1 - similarities
        
        # Average them to get the score for each patch
        # patch_scores shape: (num_patches,)
        patch_scores = np.mean(lowest_k_distances, axis=1)
        
        # Apply mask to keep only foreground patches (from main.py)
        patch_scores[~all_masks[i]] = 0.0
        
        # 5: Calculate Image-Level Score
        # Aggregate patch scores to a single image score
        # Use the mean of the 1% HIGHEST patch scores
        k_image = max(1, int(0.01 * patch_scores.shape[0]))
        
        # Partition the patch_scores array to find the k_image largest scores
        highest_k_patch_scores = np.partition(patch_scores, -k_image)[-k_image:]
        
        # Average them to get the final image-level score
        image_level_anomaly_score = np.mean(highest_k_patch_scores)
        
        image_level_scores.append(float(image_level_anomaly_score))
    
    distance_calc_time = time.time() - distance_calc_start
    timing_stats['distance_calculation'] = distance_calc_time
    print(f"  Distance calculation: {distance_calc_time:.2f}s")
    
    return np.array(image_level_scores), timing_stats


def main():
    overall_start = time.time()
    
    args = parse_args()
    dataset_path = ensure_mvtech_dataset_is_downloaded(args.root_dir)
    
    model_name = f"facebook/dinov3-vit{args.model_size}-pretrain-lvd1689m"
    print("Model", model_name)
    
    if args.device == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    model_load_start = time.time()
    model = DINOv3Wrapper(
        model_name=model_name,
        smaller_edge_size=672,
        device=device
    )
    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.2f}s\n")
    
    categories = get_categories(dataset_path)
    limit = None
    if args.test:
        categories = [args.test] 
    
    AUROCs = {}
    all_timing_stats = {}
    
    for category in tqdm(categories, desc='Processing categories'):
        category_start = time.time()
        print(f"\nEvaluating category: {category}")
        subcategories = get_subcategories()[category]
        
        # Collect all image paths and labels
        path_collection_start = time.time()
        image_paths = []
        gt_labels = []
        
        img_test_folder = f"{dataset_path}/{category}/test/"
        
        for subdir in subcategories:
            cnt = 0
            data_dir = img_test_folder + f"{subdir}"
            for img_test_nr in sorted(os.listdir(data_dir)):
                if not (img_test_nr.endswith('.png') or img_test_nr.endswith('.jpg')):
                    continue
                img_path = f"{data_dir}/{img_test_nr}"
                image_paths.append(img_path)
                gt_labels.append(subdir)
                
                if limit is not None:
                    cnt += 1
                    if cnt >= limit:
                        break
            
            if limit is not None and cnt >= limit:
                break
        
        path_collection_time = time.time() - path_collection_start
        print(f"  Path collection: {path_collection_time:.2f}s")
        
        # Load all images
        image_loading_start = time.time()
        print(f"  Loading {len(image_paths)} images...")
        images = [load_image(path) for path in tqdm(image_paths, desc="Loading images", leave=False)]
        image_loading_time = time.time() - image_loading_start
        print(f"  Image loading: {image_loading_time:.2f}s")
        
        # Run anomaly detection
        detection_start = time.time()
        use_masking = args.bg_removal[category]
        anomaly_scores, timing_stats = batched_zero_shot_anomaly_detection(images, model, use_masking)
        detection_time = time.time() - detection_start
        
        # Calculate AUROC (from main.py)
        auroc_start = time.time()
        y_true = [0 if l == "good" else 1 for l in gt_labels]
        y_scores = anomaly_scores.tolist()
        AUROCs[category] = roc_auc_score(y_true, y_scores)
        auroc_time = time.time() - auroc_start
        print(f"  AUROC calculation: {auroc_time:.2f}s")
        
        category_time = time.time() - category_start
        
        # Store timing stats for this category
        all_timing_stats[category] = {
            'total_time': category_time,
            'path_collection': path_collection_time,
            'image_loading': image_loading_time,
            'detection_time': detection_time,
            'feature_extraction': timing_stats['feature_extraction'],
            'array_conversion': timing_stats['array_conversion'],
            'distance_calculation': timing_stats['distance_calculation'],
            'auroc_calculation': auroc_time,
            'num_images': len(image_paths)
        }
        
        print(f"\nCategory: {category}, AUROC: {AUROCs[category]:.4f}")
        print(f"Total time for {category}: {category_time:.2f}s")
    
    overall_time = time.time() - overall_start
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for category, auroc in AUROCs.items():
        print(f"Category: {category}, AUROC: {auroc:.4f}")
    mean_auroc = np.mean(list(AUROCs.values()))
    print(f"Mean AUROC across categories: {mean_auroc:.4f}")
    
    # Print timing summary
    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    print(f"Model loading: {model_load_time:.2f}s")
    
    total_feature_time = sum(t['feature_extraction'] for t in all_timing_stats.values())
    total_distance_time = sum(t['distance_calculation'] for t in all_timing_stats.values())
    total_image_loading_time = sum(t['image_loading'] for t in all_timing_stats.values())
    total_images = sum(t['num_images'] for t in all_timing_stats.values())
    
    print(f"\nAcross all categories:")
    print(f"  Total images processed: {total_images}")
    print(f"  Total image loading: {total_image_loading_time:.2f}s")
    print(f"  Total feature extraction: {total_feature_time:.2f}s ({100*total_feature_time/overall_time:.1f}%)")
    print(f"  Total distance calculation: {total_distance_time:.2f}s ({100*total_distance_time/overall_time:.1f}%)")
    print(f"  Overall time: {overall_time:.2f}s")
    print(f"  Time per image: {overall_time/total_images:.2f}s")
    
    # Save results
    os.makedirs("runs/zero_shot", exist_ok=True)
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dict = {
        "AUROCs": AUROCs,
        "mean_AUROC": mean_auroc,
        "model_name": model_name,
        "date": now,
        "config": vars(args),
        "timing_stats": all_timing_stats,
        "total_time": overall_time,
        "model_load_time": model_load_time
    }
    results_path = f"runs/zero_shot/{now}.json"
    with open(results_path, 'w') as f:
        json.dump(save_dict, f, indent=4)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
