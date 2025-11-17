import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from few_shot.augment_image_with_rotations import augment_image_with_rotations


def build_memory_bank(model, reference_images, use_rotation=True,
                       normalize_features=True):
    """
    Build memory bank from reference images using sklearn NearestNeighbors.

    IMPORTANT: Following AnomalyDINO, masking should be False for memory bank!
    Masking is only applied during test-time detection, not during memory bank construction.

    Args:
        model: DINOv3Wrapper instance
        reference_images: List of images (numpy arrays or PIL Images)
        use_rotation: Whether to apply 8-way rotation augmentation
        normalize_features: Whether to L2-normalize features for cosine distance

    Returns:
        nn_index: sklearn NearestNeighbors model
        features_ref: Reference features array
        num_features: Total number of patch features in memory bank
    """
    features_ref = []

    print(f"Building memory bank from {len(reference_images)} images...")
    print(f"Rotation augmentation: {use_rotation}")

    for img_ref in tqdm(reference_images, desc="Processing reference images"):
        # Apply rotation augmentation
        if use_rotation:
            img_augmented = augment_image_with_rotations(img_ref)
        else:
            img_augmented = [img_ref]

        # Extract features from each augmented image
        for aug_img in img_augmented:
            image_tensor, grid_size = model.prepare_image(aug_img)
            features = model.extract_features(image_tensor)

            # Use ALL patches (no masking)
            features_ref.append(features)

    # Concatenate all reference features
    features_ref = np.concatenate(features_ref, axis=0).astype('float32')
    print(f"Total patches in memory bank: {features_ref.shape[0]}")

    # Normalize features for cosine distance
    if normalize_features:
        norms = np.linalg.norm(features_ref, axis=1, keepdims=True)
        features_ref = features_ref / (norms + 1e-8)

    # Create sklearn NearestNeighbors index
    # Use 'cosine' metric if normalized, otherwise 'euclidean'
    metric = 'cosine' if normalize_features else 'euclidean'
    nn_index = NearestNeighbors(
        n_neighbors=1,
        metric=metric,
        algorithm='auto',
        n_jobs=-1  # Use all CPU cores
    )
    nn_index.fit(features_ref)

    print(f"Memory bank built! Index size: {features_ref.shape[0]}")

    return nn_index, features_ref, features_ref.shape[0]
