from __future__ import print_function

import os
import os.path
import shutil
import tarfile
import urllib.request

import numpy as np
from tqdm.auto import tqdm

import cv2
import random


def load_mvtec_data(dataset_path, category, num_shots=8):
    """
    Load MVTec data for few-shot anomaly detection.

    Args:
        dataset_path: Path to MVTec dataset root
        category: Product category (e.g., 'capsule', 'bottle')
        num_shots: Number of reference images for few-shot learning

    Returns:
        reference_images: List of k-shot reference images (numpy arrays)
        test_images: List of test images
        test_labels: List of test labels (0=normal, 1=anomalous)
        test_masks: List of ground truth masks (for anomalous images)
    """
    import os
    import cv2

    # Load training images (normal samples)
    train_dir = os.path.join(dataset_path, category, 'train', 'good')
    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Select k-shot reference images
    reference_images = []
    for i in range(min(num_shots, len(train_files))):
        img_path = os.path.join(train_dir, train_files[i])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        reference_images.append(img)

    print(f"Loaded {len(reference_images)} reference images for {category}")

    # Load test images and labels
    test_dir = os.path.join(dataset_path, category, 'test')
    test_subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

    test_images = []
    test_labels = []
    test_masks = []
    test_filenames = []

    for subdir in sorted(test_subdirs):
        # Label: 0 for anomalous, 1 for normal (inverted from typical convention)
        # We'll fix this: 1 for anomalous, 0 for normal
        is_good = (subdir == 'good')
        label = 0 if is_good else 1

        test_subdir = os.path.join(test_dir, subdir)
        test_files = sorted([f for f in os.listdir(test_subdir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        for filename in test_files:
            # Load test image
            img_path = os.path.join(test_subdir, filename)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            test_images.append(img)
            test_labels.append(label)
            test_filenames.append(filename)

            # Load ground truth mask if anomalous
            if not is_good:
                mask_dir = os.path.join(dataset_path, category, 'ground_truth', subdir)
                mask_filename = filename.replace('.png', '_mask.png')
                mask_path = os.path.join(mask_dir, mask_filename)

                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    # Binarize mask
                    mask = (mask > 127).astype(np.uint8)
                else:
                    # Create empty mask if not found
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                test_masks.append(mask)
            else:
                # Normal images have no anomalies
                test_masks.append(np.zeros(img.shape[:2], dtype=np.uint8))

    print(f"Loaded {len(test_images)} test images ({sum(test_labels)} anomalous, {len(test_labels) - sum(test_labels)} normal)")

    return reference_images, test_images, test_labels, test_masks


def ensure_mvtech_dataset_is_downloaded(root_dir='.', force_download=False):
    """
    Download and extract the MVTec Anomaly Detection dataset.

    Args:
        root_dir (string): Directory where the dataset will be downloaded and extracted.
                          Default is current directory.
        force_download (bool): If True, download even if dataset already exists.

    Returns:
        str: Path to the extracted dataset directory (mvtec_anomaly_detection)
    """
    dataset_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"

    root_dir = os.path.expanduser(root_dir)
    archive_path = os.path.join(root_dir, "mvtec_anomaly_detection.tar.xz")
    extract_dir = os.path.join(root_dir, "mvtec_anomaly_detection")

    # Check if dataset already exists
    if os.path.exists(extract_dir) and not force_download:
        print(f"Dataset already exists at {extract_dir}")
        return extract_dir

    # Download the dataset
    print(f"Downloading MVTec Anomaly Detection dataset...")
    try:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading") as pbar:
            def reporthook(count, block_size, total_size):
                if pbar.total is None and total_size:
                    pbar.total = total_size
                pbar.update(block_size)

            urllib.request.urlretrieve(dataset_url, archive_path, reporthook)
        print("Download complete!")
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        if os.path.exists(archive_path):
            os.remove(archive_path)
        raise

    # Extract the archive
    print(f"Extracting archive to {root_dir}...")
    try:
        with tarfile.open(archive_path, 'r:xz') as tar:
            tar.extractall(path=extract_dir)
        print("Extraction complete!")
    except Exception as e:
        print(f"Error extracting archive: {e}")
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        raise
    finally:
        # Clean up the archive file
        if os.path.exists(archive_path):
            os.remove(archive_path)
            print(f"Removed archive file: {archive_path}")

    return extract_dir


def get_categories(dataset_dir="."):
    return [
        name
        for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name))
    ]
    
def sample_mvtec_images(
    dataset_path,
    categories=None,
    split="test",
    num_per_category=None,
    label_filter="all",
    include_masks=True,
    shuffle=True,
    seed=None,
):

    if seed is not None:
        random.seed(seed)

    # Determine which categories to use
    if not categories:
        categories = get_categories(dataset_dir=dataset_path)

    images = []
    labels = []
    masks = [] if include_masks else None
    image_categories = []
    filenames = []

    for category in sorted(categories):
        cat_images = []
        cat_labels = []
        cat_masks = [] if include_masks else None
        cat_filenames = []

        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        # ---------- TRAIN SPLIT ----------
        if split == "train":
            # Train only has 'good' (normal) images in MVTec
            if label_filter == "anomalous":
                # No anomalous samples in train split
                continue

            train_dir = os.path.join(dataset_path, category, "train", "good")
            if not os.path.isdir(train_dir):
                continue

            train_files = sorted(
                f
                for f in os.listdir(train_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )

            if shuffle:
                random.shuffle(train_files)

            if num_per_category is not None:
                train_files = train_files[:num_per_category]

            for filename in train_files:
                img_path = os.path.join(train_dir, filename)
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                cat_images.append(img)
                cat_labels.append(0)  # normal
                cat_filenames.append(filename)

                if include_masks:
                    # No ground truth masks for train split; use all-zero mask
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    cat_masks.append(mask)

        # ---------- TEST SPLIT ----------
        else:  # split == "test"
            test_dir = os.path.join(dataset_path, category, "test")
            if not os.path.isdir(test_dir):
                continue

            # Subdirectories: 'good' + defect types
            test_subdirs = [
                d
                for d in os.listdir(test_dir)
                if os.path.isdir(os.path.join(test_dir, d))
            ]

            for subdir in sorted(test_subdirs):
                is_good = (subdir == "good")
                label = 0 if is_good else 1

                # Apply label_filter
                if label_filter == "normal" and not is_good:
                    continue
                if label_filter == "anomalous" and is_good:
                    continue

                subdir_path = os.path.join(test_dir, subdir)
                files = sorted(
                    f
                    for f in os.listdir(subdir_path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                )

                if shuffle:
                    random.shuffle(files)

                # If num_per_category is set, limit total per category,
                # not per subdir. So we check length inside the loop.
                for filename in files:
                    if (
                        num_per_category is not None
                        and len(cat_images) >= num_per_category
                    ):
                        break

                    img_path = os.path.join(subdir_path, filename)
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        continue
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    cat_images.append(img)
                    cat_labels.append(label)
                    cat_filenames.append(filename)

                    if include_masks:
                        if not is_good:
                            # Load ground truth mask if anomalous
                            mask_dir = os.path.join(
                                dataset_path, category, "ground_truth", subdir
                            )
                            mask_filename = filename.replace(".png", "_mask.png")
                            mask_path = os.path.join(mask_dir, mask_filename)

                            if os.path.exists(mask_path):
                                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                                mask = (mask > 127).astype(np.uint8)
                            else:
                                # Fallback: empty mask
                                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        else:
                            # No anomalies for 'good'
                            mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        cat_masks.append(mask)

        # Append category-wise data to global lists
        images.extend(cat_images)
        labels.extend(cat_labels)
        image_categories.extend([category] * len(cat_images))
        filenames.extend(cat_filenames)

        if include_masks:
            masks.extend(cat_masks)

    print(
        f"Sampled {len(images)} images from categories={categories}, "
        f"split='{split}', label_filter='{label_filter}'"
    )

    return images, labels, masks, image_categories, filenames
