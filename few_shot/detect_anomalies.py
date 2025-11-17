import numpy as np


def detect_anomalies(model, nn_index, test_image, use_masking=True,
                     k_neighbors=1, normalize_features=True):
    """
    Detect anomalies in a test image using memory bank (sklearn version).

    Args:
        model: DINOv3Wrapper instance
        nn_index: sklearn NearestNeighbors model (memory bank)
        test_image: Test image (numpy array or PIL Image)
        use_masking: Whether to apply PCA-based masking
        k_neighbors: Number of nearest neighbors to consider
        normalize_features: Whether to L2-normalize features

    Returns:
        image_score: Image-level anomaly score
        anomaly_map: Patch-level anomaly map (grid_h, grid_w)
        patch_scores: Raw patch-level scores (num_patches,)
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

    # Find k nearest neighbors in memory bank
    distances, indices = nn_index.kneighbors(features_masked, n_neighbors=k_neighbors)

    # Average distances if k > 1
    if k_neighbors > 1:
        distances = distances.mean(axis=1)
    else:
        distances = distances.squeeze()

    # Create full patch scores (masked patches get distance, others get 0)
    patch_scores = np.zeros(features.shape[0])
    patch_scores[mask] = distances

    # Reshape to spatial grid
    anomaly_map = patch_scores.reshape(grid_size)

    # Image-level score: mean of top 1% most anomalous patches (CVaR)
    num_top = max(1, int(len(distances) * 0.01))
    image_score = np.mean(np.sort(distances)[-num_top:])

    return image_score, anomaly_map, patch_scores
