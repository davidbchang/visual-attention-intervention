import matplotlib.pyplot as plt
import numpy as np
import torch
from aim import Image, Run, Text
from sklearn.metrics import roc_auc_score

from visual_attention_intervention.base import ExperimentOutputPaths


def compute_auroc(samples: np.ndarray, labels: np.ndarray) -> float:
    samples = np.asarray(samples).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    assert samples.shape == labels.shape, f"Samples and labels must have same shape, {samples.shape} vs {labels.shape}"
    assert set(np.unique(labels)).issubset({0, 1}), "Labels must be binary"

    # AUROC is undefined if only one class is present
    if len(np.unique(labels)) < 2:
        return float("nan")

    return roc_auc_score(labels, samples)


def compute_patch_labels(
    bboxes: list[tuple[float, float, float, float]],
    image_size: tuple[int, int],
    patch_size: int,
    bbox_source_size: tuple[int, int] | None = None,  # (source_width, source_height)
    clip: bool = True,
) -> np.ndarray:
    """
    Compute binary patch-level evidence labels for an image 
    based on whether its patches overlap with any evidence region in bboxes

    Args:
        bboxes:
            List of bounding boxes (x_min, y_min, x_max, y_max) in pixel coordinates
            defined on the *source/original* image coordinate system.
        image_size:
            Target image size (W, H).
        patch_size:
            Patch size seen by the model.
        bbox_source_size:
            (source_width, source_height) that the bbox coordinates correspond to.
            - If provided: bboxes are scaled to the target (image_size, image_size).
            - If None: bboxes are assumed already in target coordinates.
        clip:
            If True, clips scaled bboxes into [0, image_size].

    Returns:
        patch_labels: a binary np.ndarray of size equal to the number of patches in the image
    """
    image_w, image_h = image_size
    num_patches_h = image_h // patch_size
    num_patches_w = image_w // patch_size
    num_patches = num_patches_h * num_patches_w
    patch_labels = np.zeros(num_patches, dtype=np.int64)

    if not bboxes:
        return patch_labels

    # Scale factors from bbox-source coords -> target coords
    if bbox_source_size is None:
        scale_x = 1.0
        scale_y = 1.0
    else:
        src_w, src_h = bbox_source_size
        if src_w <= 0 or src_h <= 0:
            raise ValueError(f"bbox_source_size must be positive, got {bbox_source_size}")
        scale_x = image_w / float(src_w)
        scale_y = image_h / float(src_h)

    eps = 1e-6  # helps avoid including adjacent patches when a boundary falls exactly on a patch edge

    for (x1, y1, x2, y2) in bboxes:
        # Scale to target coordinate system
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        # Ensure proper ordering
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)

        # Optionally clip to image bounds
        if clip:
            x_min = float(np.clip(x_min, 0.0, float(image_w)))
            x_max = float(np.clip(x_max, 0.0, float(image_w)))
            y_min = float(np.clip(y_min, 0.0, float(image_h)))
            y_max = float(np.clip(y_max, 0.0, float(image_h)))

        # Skip degenerate or empty boxes
        if x_max <= x_min + eps or y_max <= y_min + eps:
            continue

        # Compute patch index ranges overlapped by the bbox
        col_start = int(np.floor(x_min / patch_size))
        col_end   = int(np.ceil((x_max - eps) / patch_size))
        row_start = int(np.floor(y_min / patch_size))
        row_end   = int(np.ceil((y_max - eps) / patch_size))

        # Clip patch indices
        col_start = max(col_start, 0)
        row_start = max(row_start, 0)
        col_end   = min(col_end, num_patches_w - 1)
        row_end   = min(row_end, num_patches_h - 1)

        if col_end < col_start or row_end < row_start:
            continue

        # Mark all overlapped patches as evidence
        for r in range(row_start, row_end + 1):
            start = r * num_patches_w + col_start
            end = r * num_patches_w + col_end + 1
            patch_labels[start:end] = 1

    return patch_labels


def denoise_attention_mask(
    patch_scores: np.ndarray | list,
    grid_hw: tuple[int, int] | None = None,
    lam: float = 10.0,
    return_grid: bool = False,
) -> np.ndarray:
    """
    Implements Attention Mask Denoising based on the following:

    Given patch evidence scores e (flattened) over an HxW patch grid, for each patch (i,j)
    with neighborhood N(i,j) = 3x3 window excluding (i,j), update:

        e'_{i,j} = mean_{(p,q) in N(i,j)} e_{p,q}   if  e_{i,j} > lam * max_{(p,q) in N(i,j)} e_{p,q}
                 = e_{i,j}                         otherwise

    Args:
        patch_scores: Patch evidence score vector of shape (m,) (e.g., 256) or a grid (H,W).
        grid_hw: (H, W). If None and patch_scores is 1D, infers a square grid from m.
        lam: Multiplicative threshold λ controlling strictness (paper uses λ = 10).
        return_grid: If True, return shape (H,W). Otherwise return flattened shape (m,).

    Returns:
        Denoised patch scores, same shape as input (flattened by default).
    """
    e = np.asarray(patch_scores, dtype=np.float32)

    # Accept either flattened (m,) or grid (H,W)
    if e.ndim == 2:
        H, W = e.shape
        grid = e
    elif e.ndim == 1:
        m = e.size
        if grid_hw is None:
            s = int(round(np.sqrt(m)))
            if s * s != m:
                raise ValueError(
                    f"patch_scores has length {m}, which is not a perfect square. "
                    f"Please pass grid_hw=(H,W)."
                )
            H, W = s, s
        else:
            H, W = grid_hw
            if H * W != m:
                raise ValueError(f"grid_hw {grid_hw} implies {H*W} patches, but got {m}.")
        grid = e.reshape(H, W)
    else:
        raise ValueError(f"patch_scores must be 1D or 2D, got shape {e.shape}.")

    out = grid.copy()  # will hold e'
    orig = grid        # keep original for comparisons/statistics

    for i in range(H):
        i0, i1 = max(0, i - 1), min(H, i + 2)  # [i-1, i, i+1]
        for j in range(W):
            j0, j1 = max(0, j - 1), min(W, j + 2)  # [j-1, j, j+1]

            # Extract 3x3 (or smaller at borders) neighborhood
            nb = orig[i0:i1, j0:j1].copy()

            # Remove center element (i,j) from neighborhood
            ci, cj = i - i0, j - j0
            nb_flat = nb.reshape(-1)
            center_idx = ci * (j1 - j0) + cj
            if nb_flat.size <= 1:
                # No neighbors exist (e.g., 1x1 grid)
                continue

            nb_excl = np.delete(nb_flat, center_idx)

            max_nb = float(nb_excl.max())
            mean_nb = float(nb_excl.mean())

            if float(orig[i, j]) > lam * max_nb:
                out[i, j] = mean_nb

    if return_grid:
        return out

    return out.reshape(-1)


def smooth_highlight_mask(
    e_denoised: np.ndarray | list,
    grid_hw: tuple[int, int] | None = None,
    *,
    sigma: float | None = None,               # std dev (in pixels or in grid units; see sigma_space)
    image_short_side: int,                    # shorter side of the image in pixels
    patch_size: int,                          # patch size in pixels (used to convert pixels -> grid units)
    smooth_strength: float | None = None,     # in the range [0,1]
    sigma_space: str = "pixels",              # "pixels" or "grid"
    kernel_size: int | None = None,           # optional explicit odd kernel size k
    padding: str = "reflect",                 # "reflect", "edge", or "constant"
    return_grid: bool = False,
) -> np.ndarray:
    """
    Implements Gaussian smoothing of the denoised attention mask e'

    Args:
        e_denoised: e' as a flat vector (m,) (e.g., 256) or a grid (H,W).
        grid_hw: (H,W) if e_denoised is 1D and not a perfect square.
        sigma: Gaussian std dev. If None, computed as smooth_strength * image_short_side.
        image_short_side: shorter image side in pixels (used for sigma if sigma is None).
        patch_size: patch size in pixels (used to convert pixel sigma to grid sigma).
        smooth_strength: hyperparameter in [0,1]. Used only if sigma is None.
        sigma_space: "pixels" => sigma is in pixels and converted to patch-grid units via /patch_size.
                     "grid"   => sigma is already in patch-grid units.
        kernel_size: optional explicit odd k for the Gaussian kernel. If None, uses k = 2*ceil(3*sigma_grid)+1.
        padding: boundary handling for convolution ("reflect", "edge", "constant").
        return_grid: if True returns (H,W); otherwise returns flattened (m,).

    Returns:
        e_smoothed (same shape convention as input; flattened by default).
    """
    e = np.asarray(e_denoised, dtype=np.float32)

    # ---- reshape to grid ----
    if e.ndim == 2:
        H, W = e.shape
        grid = e
    elif e.ndim == 1:
        m = e.size
        if grid_hw is None:
            s = int(round(np.sqrt(m)))
            if s * s != m:
                raise ValueError(
                    f"e_denoised has length {m}, not a perfect square. Please pass grid_hw=(H,W)."
                )
            H, W = s, s
        else:
            H, W = grid_hw
            if H * W != m:
                raise ValueError(f"grid_hw={grid_hw} implies {H*W} elements, but got {m}.")
        grid = e.reshape(H, W)
    else:
        raise ValueError(f"e_denoised must be 1D or 2D, got shape {e.shape}.")

    # ---- determine sigma ----
    if sigma is None:
        if smooth_strength is None:
            raise ValueError("Provide either sigma or smooth_strength (with image_short_side).")
        sigma_eff = float(smooth_strength) * float(image_short_side)
    else:
        sigma_eff = float(sigma)

    if sigma_eff <= 0:
        # No smoothing
        return grid if (return_grid or e.ndim == 2) else grid.reshape(-1)

    if sigma_space not in {"pixels", "grid"}:
        raise ValueError("sigma_space must be 'pixels' or 'grid'.")

    sigma_grid = sigma_eff / float(patch_size) if sigma_space == "pixels" else sigma_eff

    if sigma_grid <= 0:
        return grid if (return_grid or e.ndim == 2) else grid.reshape(-1)

    # ---- build Gaussian kernel G_sigma ----
    if kernel_size is None:
        r = int(np.ceil(3.0 * sigma_grid))
        kernel_size = 2 * r + 1
    else:
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        r = kernel_size // 2

    xs = np.arange(-r, r + 1, dtype=np.float32)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    G = np.exp(-(xx * xx + yy * yy) / (2.0 * (sigma_grid * sigma_grid)))
    G /= (G.sum() + 1e-12)  # ensure sum_{p,q} G(p,q) = 1

    # ---- pad grid ----
    pad_mode = padding
    if pad_mode not in {"reflect", "edge", "constant"}:
        raise ValueError("padding must be one of: 'reflect', 'edge', 'constant'.")

    padded = np.pad(grid, ((r, r), (r, r)), mode=pad_mode)

    # ---- convolution (same output size) ----
    out = np.empty_like(grid, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            window = padded[i:i + kernel_size, j:j + kernel_size]
            out[i, j] = float(np.sum(window * G))

    # ---- return with requested shape ----
    if return_grid or e.ndim == 2:
        return out

    return out.reshape(-1)


def qwen_data_collator(examples):
    images = []
    image_names = []
    prompts = []
    labels = []
    image_links = []
    entities = []

    for example in examples:
        images.append(example["image"])
        image_names.append(example["image_name"])
        image_links.append(example["image_link"])
        if example["entities"]:
            entities.append(example["entities"])

        prompt = (
            "Look at the image and decide whether the statement is True or False.\n"
            f"Statement: {example["caption"]}\n"
            "Answer with exactly one word: True or False."
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]

        prompts.append(messages)
        labels.append("True" if example["label"] else "False")

    entities = entities if len(entities) > 0 else None

    return prompts, labels, images, image_names, image_links, entities


def save_attention_image_overlay(img, attn_heatmap, output_file_path, alpha=0.5):
    fig, _ = plt.subplots()
    plt.imshow(img)
    plt.imshow(attn_heatmap, alpha=alpha)
    plt.axis('off')
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close(fig)


def plot_layer_attentions(output_layer_attentions, image_name, output_file_path, weight=None):
    layer_image_attention_weights = []
    layer_text_attention_weights = []

    for layer_attentions in output_layer_attentions:
        image_attention_weights = layer_attentions.image_attention_weights
        text_attention_weights = layer_attentions.text_attention_weights

        # get sum of attention weights averaged across all heads
        image_attention_weight_sum = torch.sum(torch.mean(image_attention_weights, dim=1))
        text_attention_weight_sum = torch.sum(torch.mean(text_attention_weights, dim=1))

        layer_image_attention_weights.append(image_attention_weight_sum)
        layer_text_attention_weights.append(text_attention_weight_sum)

    layer_indices = list(range(len(output_layer_attentions)))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Sum of Attention Weights Given to Image Tokens', color=color)
    ax1.plot(layer_indices, layer_image_attention_weights, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Sum of Attention Weights Given to Text Tokens', color=color)
    ax2.plot(layer_indices, layer_text_attention_weights, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    title = f"Comparison of Attention Weights Allocated to Image vs Text Tokens\nImage Name: {image_name}"
    if weight is not None:
        title += f", Image Weight: {weight}"
    plt.title(title)

    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.close(fig)


def track_aim_run(
    aim_run: Run,
    experiment_output_paths: ExperimentOutputPaths, 
    prediction_result: dict[str, any], 
    step: int, 
    context: dict[str, any]
):
    file_name = prediction_result["image_filename"]
    prompt = prediction_result["prompt"]
    label = prediction_result["label"]
    prediction = prediction_result["prediction"]

    aim_image_attn_plots = Image(str(experiment_output_paths.attention_plots_dir_path / file_name), caption=file_name, format='jpeg', optimize=True)
    aim_image_attn_weights = Image(str(experiment_output_paths.attention_weights_dir_path / file_name), caption=file_name, format='jpeg', optimize=True)
    aim_image_heatmap = Image(str(experiment_output_paths.heatmaps_dir_path / file_name), caption=file_name, format='jpeg', optimize=True)

    aim_run.track(aim_image_attn_plots, name=experiment_output_paths.attention_plots_dir_name, step=step, context=context)
    aim_run.track(aim_image_attn_weights, name=experiment_output_paths.attention_weights_dir_name, step=step, context=context)
    aim_run.track(aim_image_heatmap, name=experiment_output_paths.heatmaps_dir_name, step=step, context=context)

    aim_run.track(Text(prompt), name="Caption", step=step, context=context)
    aim_run.track(Text(label), name="Label", step=step, context=context)
    aim_run.track(Text(prediction), name="Prediction", step=step, context=context)
