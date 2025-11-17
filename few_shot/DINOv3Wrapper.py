import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from transformers import AutoImageProcessor, AutoModel


class DINOv3Wrapper:
    """Wrapper for DINOv3 model for anomaly detection with patch-level features."""

    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
                 device=None, smaller_edge_size=518):
        """
        Initialize DINOv3 wrapper.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on (auto-detect if None)
            smaller_edge_size: Target size for the smaller edge (increases patch resolution)
                             Default 518 gives ~32x32 grid for vits16
                             Use 640 for ~40x40 grid (matches AnomalyDINO paper)
        """
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size

        # Auto-detect device
        # NOTE: MPS has compatibility issues with FAISS, using CPU for stability
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                # Force CPU even if MPS is available (FAISS compatibility)
                self.device = torch.device("cpu")
        else:
            self.device = device

        print(f"Loading {model_name} on {self.device}...")

        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        # Get model config
        self.num_register_tokens = self.model.config.num_register_tokens
        self.hidden_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size

        print(f"Model loaded! Hidden dim: {self.hidden_dim}, Register tokens: {self.num_register_tokens}, Patch size: {self.patch_size}")
        print(f"Smaller edge size: {self.smaller_edge_size}")

    def prepare_image(self, image):
        """
        Prepare image for feature extraction using AnomalyDINO's resizing strategy.
        Resizes by smaller edge and crops to multiples of patch size.

        Args:
            image: PIL Image or numpy array (H, W, C)

        Returns:
            image_tensor: Processed image tensor
            grid_size: Tuple of (grid_h, grid_w)
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        # Resize by smaller edge (preserves aspect ratio)
        w, h = image.size
        if h < w:
            new_h = self.smaller_edge_size
            new_w = int(w * (self.smaller_edge_size / h))
        else:
            new_w = self.smaller_edge_size
            new_h = int(h * (self.smaller_edge_size / w))

        image = image.resize((new_w, new_h), Image.BICUBIC)

        # Process with AutoImageProcessor (this normalizes)
        inputs = self.processor(images=image, return_tensors="pt", do_resize=False)
        image_tensor = inputs['pixel_values'].to(self.device)

        # Crop to dimensions that are multiples of patch size
        _, _, height, width = image_tensor.shape  # B, C, H, W
        cropped_height = height - (height % self.patch_size)
        cropped_width = width - (width % self.patch_size)
        image_tensor = image_tensor[:, :, :cropped_height, :cropped_width]

        # Calculate grid size
        grid_h = cropped_height // self.patch_size
        grid_w = cropped_width // self.patch_size
        grid_size = (grid_h, grid_w)

        return image_tensor, grid_size

    def extract_features(self, image_tensor):
        """
        Extract patch-level features from image tensor.

        Args:
            image_tensor: Preprocessed image tensor

        Returns:
            features: Numpy array of patch features (num_patches, hidden_dim)
        """
        with torch.inference_mode():
            outputs = self.model(pixel_values=image_tensor)
            last_hidden = outputs.last_hidden_state  # (batch, num_tokens, hidden_dim)

            # Skip CLS token (1) + register tokens (4) to get patch features
            patch_features = last_hidden[:, 1 + self.num_register_tokens:, :]

            # Move to CPU immediately and convert to numpy
            features = patch_features.squeeze(0).cpu().numpy().astype('float32', copy=True)

            # Clear GPU/MPS memory
            del outputs, last_hidden, patch_features
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return features

    def compute_background_mask(self, img_features, grid_size, threshold_percentile=70,
                                 masking_type=False, kernel_size=3, border=0.2,
                                 smoothing_sigma=4, n_components=1, aggregation='first'):
        """
        PCA-based foreground segmentation using percentile thresholding.

        Args:
            img_features: Patch features (num_patches, hidden_dim)
            grid_size: Tuple of (grid_h, grid_w)
            threshold_percentile: Percentile for thresholding (default 70 = keep top 30%)
            masking_type: Boolean, whether to apply masking
            kernel_size: Size of kernel for morphological operations (should be odd)
            border: Fraction of border region to check for adaptive flipping
            smoothing_sigma: Gaussian smoothing sigma for the PCA map
            n_components: Number of PCA components to use (default 1)
            aggregation: How to combine multiple components:
                        'first' - use only first component (default, AnomalyDINO approach)
                        'l2' - L2 norm across components
                        'mean' - Mean across components
                        'max' - Max across components

        Returns:
            mask: Boolean mask
        """
        if not masking_type:
            return np.ones(img_features.shape[0], dtype=bool)

        # Perform PCA with multiple components
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
        pca_components = pca.fit_transform(img_features.astype(np.float32))

        # Aggregate multiple components into single saliency map
        if n_components == 1 or aggregation == 'first':
            # Use only first component (standard approach)
            pca_map = pca_components[:, 0] if pca_components.ndim > 1 else pca_components
        elif aggregation == 'l2':
            # L2 norm across components
            pca_map = np.linalg.norm(pca_components, axis=1)
        elif aggregation == 'mean':
            # Mean across components
            pca_map = np.mean(np.abs(pca_components), axis=1)
        elif aggregation == 'max':
            # Max absolute value across components
            pca_map = np.max(np.abs(pca_components), axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        # Reshape to spatial grid
        pca_map = pca_map.reshape(grid_size)

        # Apply Gaussian smoothing to reduce noise
        pca_map_smooth = gaussian_filter(pca_map, sigma=smoothing_sigma)

        # Apply percentile-based threshold
        threshold = np.percentile(pca_map_smooth, threshold_percentile)
        mask = pca_map_smooth > threshold

        # Adaptive: check if center region is kept, flip if not
        h, w = grid_size
        center_region = mask[int(h * border):int(h * (1-border)),
                            int(w * border):int(w * (1-border))]

        if center_region.size > 0 and center_region.sum() <= center_region.size * 0.35:
            # Flip mask
            mask = pca_map_smooth < threshold

        # Morphological post-processing
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask.flatten().astype(bool)

    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens