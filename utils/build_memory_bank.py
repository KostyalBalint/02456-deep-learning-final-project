import numpy as np
import hnswlib
from tqdm.auto import tqdm
from utils.augment_image_with_rotations import augment_image_with_rotations


def build_memory_bank(model, reference_images, use_rotation=True,
                       normalize_features=True):

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

    # Normalize features for cosine distance
    if normalize_features:
        norms = np.linalg.norm(features_ref, axis=1, keepdims=True)
        features_ref = features_ref / (norms + 1e-8)

    # Create hnswlib index
    # For normalized vectors, 'ip' (inner product) gives cosine similarity
    dim = features_ref.shape[1]
    num_elements = features_ref.shape[0]

    # Create HNSW index with inner product space
    nn_index = hnswlib.Index(space='ip', dim=dim)

    # Initialize index
    # M = number of connections per element (higher = more accurate, more memory)
    # ef_construction = size of dynamic candidate list during construction
    nn_index.init_index(max_elements=num_elements, ef_construction=200, M=16)

    # Add vectors to index
    nn_index.add_items(features_ref)

    # Set ef parameter for search (higher = more accurate but slower)
    nn_index.set_ef(50)

    print(f"Memory bank built! Index size: {num_elements}, hnswlib HNSW index")

    return nn_index, features_ref, features_ref.shape[0]
