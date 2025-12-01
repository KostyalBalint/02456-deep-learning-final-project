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

def calculate_features(model: DINOv3Wrapper, test_image: Image.Image, use_masking=True):
    # Extract features from test image
    image_tensor, grid_size = model.prepare_image(test_image)
    features = model.extract_features(image_tensor)

    # Apply PCA masking with percentile threshold
    if use_masking:
        mask = model.compute_background_mask(features, grid_size,
                                            threshold_percentile=70,
                                            masking_type=True,
                                            smoothing_sigma=4)
    else:
        mask = np.ones(features.shape[0], dtype=bool)

    # Normalize features
    return features[mask].astype('float32')


def dists_to_score(patch_scores: List[np.ndarray]):
    # 5: Calculate Image-Level Score
    # Aggregate patch scores to a single image score
    # Use the mean of the 1% HIGHEST patch scores 
    k_image = max(1, int(0.01 * patch_scores.shape[0]))
    
    # Partition the patch_scores array to find the k_image largest scores
    highest_k_patch_scores = np.partition(patch_scores, -k_image)[-k_image:]
    
    # Average them to get the final image-level score
    return np.mean(highest_k_patch_scores)

def calculate_cosine_distances_2(all_features, test_idx, device=torch.device("mps"), quantile = 0.001, batch_size=50):
    """Process in smaller batches to avoid OOM errors."""
    with torch.no_grad():
        # Only convert test features to tensor
        sample_features = torch.tensor(all_features[test_idx], device=device, dtype=torch.float32)
        normalized_sample = torch.nn.functional.normalize(sample_features, dim=1)
        
        num_patches = sample_features.shape[0]
        k = max(1, int(quantile * sum(f.shape[0] for i, f in enumerate(all_features) if i != test_idx)))
        
        # Store top-k distances per patch
        top_k_distances = torch.full((num_patches, k), float('inf'), device=device)
        
        # Process reference features in batches
        for i, ref_features in enumerate(all_features):
            if i == test_idx:
                continue
                
            ref_tensor = torch.tensor(ref_features, device=device, dtype=torch.float32)
            normalized_ref = torch.nn.functional.normalize(ref_tensor, dim=1)
            
            # Compute distances for this reference batch
            cosine_similarity = torch.mm(normalized_sample, normalized_ref.t())
            cosine_distance = 1 - cosine_similarity
            
            # Merge with existing top-k
            combined = torch.cat([top_k_distances, cosine_distance], dim=1)
            top_k_distances = torch.topk(combined, k, dim=1, largest=False).values
            
            # Clean up
            del ref_tensor, normalized_ref, cosine_similarity, cosine_distance, combined
            if device.type == 'mps':
                torch.mps.empty_cache()
        
        # Average the top-k smallest distances for each patch
        means_below_quantile = top_k_distances.mean(dim=1)
        result = means_below_quantile.cpu().numpy()
        
        # Final cleanup
        del sample_features, normalized_sample, top_k_distances, means_below_quantile
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
            
    return result


def calculate_cosine_distances(all_features: List[np.ndarray], test_idx: int):
        # 1. Select the patches for the current test image
        test_patches = all_features[test_idx, :, :]  # Shape: (num_patches, hidden_dim)
        
        # 2. Create the memory bank from ALL OTHER images
        filter_i = np.arange(len(all_features)) != test_idx
        memory_bank = all_features[filter_i]
        
        # Reshape bank to ( (num_images - 1) * num_patches, hidden_dim )
        memory_bank = memory_bank.reshape(-1, memory_bank.shape[-1])

        # 3. Calculate cosine distances (1 - similarity)
        # We want a matrix of shape (num_test_patches, num_bank_patches)
        
        # Normalize vectors for cosine similarity calculation
        test_patches_norm = test_patches / (np.linalg.norm(test_patches, axis=1, keepdims=True) + 1e-10)
        memory_bank_norm = memory_bank / (np.linalg.norm(memory_bank, axis=1, keepdims=True) + 1e-10)
        
        # Similarities shape: (num_patches, num_bank_patches)
        similarities = np.dot(test_patches_norm, memory_bank_norm.T)
        
        # Distances shape: (num_patches, num_bank_patches)
        distances = 1 - similarities

        # 4: Calculate Patch-Level Scores
        # For EACH test patch, find the mean of its 0.1% lowest distances to the bank 
        
        k_patch = max(1, int(0.001 * memory_bank.shape[0]))
        
        # Partition each ROW (axis=1) to find the k_patch smallest distances
        partitioned_distances = np.partition(distances, k_patch, axis=1)
        
        # Select the k_patch smallest distances for each patch
        lowest_k_distances = partitioned_distances[:, :k_patch]
        
        # Average them to get the score for each patch
        # patch_scores shape: (num_patches,)
        patch_scores = np.mean(lowest_k_distances, axis=1)

        return patch_scores

def load_image(image_path: str) -> Image.Image:
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def batched_zero_shot_anomaly_detection(dataset_path, model: DINOv3Wrapper, bg_removal:dict, test:bool):
    categories = get_categories(dataset_path)
    limit = None
    if test: 
        categories = categories[:1]  # For testing purposes, only run on the first category
        limit = 2

    AUROCs = {}

    for category in categories:
        print(f"Evaluating category: {category}")
        subcategories = get_subcategories()[category]

        # Don't store images, only features and metadata
        features_all = []
        masks_ref = []
        gt_label = []

        img_test_folder = f"{dataset_path}/{category}/test/"

        with torch.inference_mode():
            for subdir in subcategories:
                cnt = 0
                data_dir = img_test_folder + f"{subdir}"
                for img_test_nr in tqdm(sorted(os.listdir(data_dir)), desc=f"Load test set: {category} ({subdir})", leave=False):
                    img_path = f"{data_dir}/{img_test_nr}"
                    img = load_image(img_path)
                    image_tensor, grid_size = model.prepare_image(img)
                    features = model.extract_features(image_tensor)
                    mask_test = model.compute_background_mask(features, grid_size, masking_type=bg_removal[category])
                    
                    # Only keep features, not raw images
                    features_all.append(features)
                    masks_ref.append(mask_test)
                    gt_label.append(subdir)
                    
                    # Clean up
                    del img, image_tensor
                    if model.device.type == 'mps':
                        torch.mps.empty_cache()
                    elif model.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    if limit is not None:
                        cnt += 1
                        if cnt >= limit:
                            break

            # Convert list to 3D numpy array for efficient indexing
            features_all = np.array(features_all)
            
            test_dists = []
            for i in tqdm(range(len(features_all)), desc=f"Calculate distances: {category} ", leave=False):
                # Use the fast numpy-based function
                patch_scores = calculate_cosine_distances(features_all, i)
                # Apply mask to keep only foreground patches
                patch_scores[~masks_ref[i]] = 0.0
                test_dists.append(patch_scores)
            
            y_true = [0 if l == "good" else 1 for l in gt_label]
            y_scores = [dists_to_score(d) for d in test_dists]
            AUROCs[category] = roc_auc_score(y_true, y_scores)
            
            # Clean up category data
            del features_all, masks_ref, test_dists
            if model.device.type == 'mps':
                torch.mps.empty_cache()
            elif model.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return AUROCs


if __name__ == "__main__":
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

    model = DINOv3Wrapper(
        model_name=model_name,
        smaller_edge_size=1024,
        device=device
    )

    AUROCs = batched_zero_shot_anomaly_detection(dataset_path, model, bg_removal=args.bg_removal, test=args.test)

    for category, auroc in AUROCs.items():
        print(f"Category: {category}, AUROC: {auroc:.4f}")
    mean_auroc = np.mean(list(AUROCs.values()))
    print(f"Mean AUROC across categories: {mean_auroc:.4f}")

    # save results to test_scores/zero_shot/model_name.json
    os.makedirs("runs/zero_shot", exist_ok=True)
    # get date and time
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dict = {
        "AUROCs": AUROCs,
        "mean_AUROC": mean_auroc,
        "model_name": model_name,
        "date": now,
        "config": vars(args)
    }
    results_path = f"runs/zero_shot/{now}.json"
    with open(results_path, 'w') as f:
        json.dump(save_dict, f, indent=4)
    print(f"Saved results to {results_path}")