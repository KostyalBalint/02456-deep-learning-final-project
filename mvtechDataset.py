from __future__ import print_function

import os
import os.path
import shutil
import tarfile
import urllib.request

import numpy as np
from tqdm.auto import tqdm


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