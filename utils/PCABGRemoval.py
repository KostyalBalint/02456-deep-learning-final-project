import numpy as np
import cv2
from sklearn.decomposition import PCA


def get_foreground_mask_pca(
    feats: np.ndarray,
    grid_size: tuple[int, int],
    min_fg_ratio: float = 0.03,
    max_fg_ratio: float = 0.7,
    debug: bool = False,
    return_debug: bool = False,
):
    H_grid, W_grid = grid_size
    assert feats.shape[0] == H_grid * W_grid, "feats and grid_size mismatch"

    # PCA (first component) on patch embeddings
    pca = PCA(n_components=1, svd_solver="randomized", random_state=42)
    pc_vals = pca.fit_transform(feats.astype(np.float32)).flatten()

    pc_mean = pc_vals.mean()
    pc_std = pc_vals.std() + 1e-8
    pc_z = (pc_vals - pc_mean) / pc_std

    # Map z-scores to [0, 1] then [0,255] for Otsu
    pc_min, pc_max = pc_z.min(), pc_z.max()
    pc_01 = (pc_z - pc_min) / (pc_max - pc_min + 1e-8)
    pc_8u = (pc_01 * 255).astype(np.uint8)

    # Otsu threshold on normalized PC
    otsu_thr_8u, _ = cv2.threshold(
        pc_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thr_01 = otsu_thr_8u / 255.0
    thr_z = pc_min + thr_01 * (pc_max - pc_min + 1e-8)

    mask_otsu = pc_01 >= thr_01
    mask_otsu_2d = mask_otsu.reshape(H_grid, W_grid)

    # Optional flip: ensure center is mostly foreground
    h0, h1 = int(H_grid * 0.25), int(H_grid * 0.75)
    w0, w1 = int(W_grid * 0.25), int(W_grid * 0.75)
    center = mask_otsu_2d[h0:h1, w0:w1]
    center_ratio = center.mean() if center.size > 0 else 0.0

    mask_center = mask_otsu_2d.copy()
    if center_ratio < 0.35:  # assume object is central
        mask_center = ~mask_center

    # Enforce reasonable foreground ratio via fallback if needed
    fg_ratio_center = mask_center.mean()
    mask_fg = mask_center.copy()
    thr_fallback_z = None

    if fg_ratio_center < min_fg_ratio or fg_ratio_center > max_fg_ratio:
        thr_fallback_z = np.percentile(pc_z, 80)  # top 20% as foreground
        mask_fg = (pc_z >= thr_fallback_z).reshape(H_grid, W_grid)

    fg_ratio_pre_morph = mask_fg.mean()

    # Morphological cleanup + largest connected component
    mask_uint8 = mask_fg.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels = cv2.connectedComponents(mask_clean)
    if num_labels > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0  # ignore background label
        max_label = sizes.argmax()
        mask_clean = (labels == max_label).astype(np.uint8)

    fg_ratio_final = mask_clean.mean()

    # Detect "texture / bad segmentation" using *raw* PC variance
    # AND check if PCA map is mostly a simple plane (global gradient).

 
    # gradient-like (planar) structure on 2D PCA map 
    pc2d = pc_z.reshape(H_grid, W_grid)

    ys, xs = np.mgrid[0:H_grid, 0:W_grid]
    A = np.stack([xs.ravel(), ys.ravel(), np.ones_like(xs).ravel()], axis=1)
    coef, _, _, _ = np.linalg.lstsq(A, pc2d.ravel(), rcond=None)
    plane = (A @ coef).reshape(H_grid, W_grid)

    residual = pc2d - plane
    res_std = residual.std()
    total_std = pc2d.std() + 1e-8
    planar_ratio = res_std / total_std  # how much variance is NOT explained by a plane

    grad_resid_thresh = 0.4  # tune: lower -> more maps considered "pure gradient"
    gradient_like = planar_ratio < grad_resid_thresh

    # geometric thin-mask check (like before) 
    ys_fg, xs_fg = np.where(mask_clean > 0)
    if ys_fg.size > 0:
        h_span = (ys_fg.max() - ys_fg.min() + 1) / H_grid
        w_span = (xs_fg.max() - xs_fg.min() + 1) / W_grid
    else:
        h_span, w_span = 0.0, 0.0

    very_slender = min(h_span, w_span) < 0.35


    if gradient_like or very_slender:
        mask_clean = np.ones((H_grid, W_grid), dtype=np.uint8)
        fg_ratio_final = 1.0


    if debug:
        print(
            f"Otsu thr(z)={thr_z:.3f}, center_ratio={center_ratio:.2f}, "
            f"fg_center={fg_ratio_center:.2f}, fg_pre={fg_ratio_pre_morph:.2f}, "
            f"fg_final={fg_ratio_final:.2f}, h_span={h_span:.2f}, w_span={w_span:.2f}, "
            f"very_slender={very_slender}"
        )
        if thr_fallback_z is not None:
            print(f"Fallback threshold used (z)={thr_fallback_z:.3f}")

    pc_map_grid = pc_z.reshape(H_grid, W_grid)

    if not return_debug:
        return mask_clean, pc_map_grid

    debug_info = {
        "pc_z": pc_z.reshape(H_grid, W_grid),
        "pc_01": pc_01.reshape(H_grid, W_grid),
        "thr_z": thr_z,
        "thr_fallback_z": thr_fallback_z,
        "mask_otsu": mask_otsu_2d.astype(np.uint8),
        "mask_center": mask_center.astype(np.uint8),
        "mask_fg_before_morph": mask_fg.astype(np.uint8),
        "mask_clean": mask_clean.astype(np.uint8),
        "center_ratio": center_ratio,
        "fg_ratio_center": fg_ratio_center,
        "fg_ratio_pre": fg_ratio_pre_morph,
        "fg_ratio_final": fg_ratio_final,
        "h_span": h_span,
        "w_span": w_span,
        "gradient_like": gradient_like,
        "very_slender": very_slender,
    }

    return mask_clean, pc_map_grid, debug_info
