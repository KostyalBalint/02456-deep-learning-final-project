import numpy as np
from sklearn.neighbors import NearestNeighbors

def detect_anomalies(model, nn_index: NearestNeighbors, test_image, use_masking=True, normalize_features=True):
    """
    Zero-shot anomaly detection using batched memory bank approach.
    
    Args:
        model: DINOv3Wrapper instance
        nn_index: NearestNeighbors index containing the memory bank (all patches from all test images)
        test_image: PIL Image to detect anomalies in
        use_masking: Whether to apply background masking
        normalize_features: Whether to normalize features before distance calculation
    
    Returns:
        image_score: Image-level anomaly score
        anomaly_map: Spatial anomaly map (grid_size shape)
        patch_scores: Patch-level anomaly scores (flattened)
    """
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
    features_masked = features[mask].astype('float32')
    if normalize_features:
        norms = np.linalg.norm(features_masked, axis=1, keepdims=True)
        features_masked = features_masked / (norms + 1e-8)

    # Calculate Patch-Level Scores
    # For EACH test patch, find the mean of its 0.1% lowest distances to the memory bank
    # Get the memory bank size from the nn_index
    memory_bank_size = nn_index.n_samples_fit_
    k_patch = max(1, int(0.001 * memory_bank_size))

    # Find k nearest neighbors (lowest distances) in memory bank
    # distances shape: (num_masked_patches, k_patch)
    # The distances are cosine distances (1 - cosine_similarity) since features are normalized
    distances, indices = nn_index.kneighbors(features_masked, n_neighbors=k_patch)

    # Average the k lowest distances for each patch
    # This gives the anomaly score for each masked patch
    patch_scores_masked = np.mean(distances, axis=1)

    # Create full patch scores (masked patches get their scores, others get 0)
    patch_scores = np.zeros(features.shape[0])
    patch_scores[mask] = patch_scores_masked

    # Reshape to spatial grid for anomaly map
    anomaly_map = patch_scores.reshape(grid_size)

    # Calculate Image-Level Score
    # Use the mean of the top 1% most anomalous patches (CVaR approach)
    # Only consider masked patches for image-level score
    num_top = max(1, int(len(patch_scores_masked) * 0.01))
    image_score = np.mean(np.sort(patch_scores_masked)[-num_top:])

    return image_score, anomaly_map, patch_scores
