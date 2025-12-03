import numpy as np
import hnswlib
from utils.PCABGRemoval import get_foreground_mask_pca

def detect_anomalies(model, nn_index: hnswlib.Index, test_image, use_masking=True, masking_method='threshold', normalize_features=True):

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

    # Calculate Patch-Level Scores
    memory_bank_size = nn_index.get_current_count()
    k_patch = max(1, int(0.001 * memory_bank_size))

    # Find k nearest neighbors using hnswlib
    indices, distances = nn_index.knn_query(features_masked, k=k_patch)

    # Average the k lowest distances for each patch
    patch_scores_masked = np.mean(distances, axis=1)

    # Create full patch scores
    patch_scores = np.zeros(features.shape[0])
    patch_scores[mask] = patch_scores_masked

    # Reshape to spatial grid for anomaly map
    anomaly_map = patch_scores.reshape(grid_size)

    # Calculate Image-Level Score
    num_top = max(1, int(len(patch_scores_masked) * 0.01))
    image_score = np.mean(np.sort(patch_scores_masked)[-num_top:])

    return image_score, anomaly_map, patch_scores, mask_grid
