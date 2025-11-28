import numpy as np


def detect_anomalies(model, nn_index, test_image, use_masking=True, masking_method='threshold',
                     k_neighbors=1, normalize_features=True):

    # Extract features from test image
    image_tensor, grid_size = model.prepare_image(test_image)
    features = model.extract_features(image_tensor)

    # Apply PCA masking with percentile threshold
    if use_masking:
        if masking_method == 'threshold':
            mask = model.compute_background_mask(features, grid_size,
                                                threshold_percentile=70,
                                                masking_type=False,
                                                smoothing_sigma=4)
        if masking_method == 'pca':
            mask_grid, pc_map, dbg = get_foreground_mask_pca(
            feats=features,
            grid_size=grid_size,
            debug=False,
            return_debug=True,
            )       
            mask = (mask_grid.reshape(-1) > 0)
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
