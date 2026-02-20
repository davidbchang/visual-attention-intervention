import json
import hashlib
import random
import threading
import traceback
from pathlib import Path
from functools import lru_cache

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, set_seed

from visual_attention_intervention.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)
from visual_attention_intervention.utils import (
    denoise_attention_mask,
    save_attention_image_overlay,
    smooth_highlight_mask,
)

if Path("/data").exists():
    import os
    os.environ.setdefault("HF_HOME", "/data/.huggingface")

torch.manual_seed(42)
set_seed(42)

CACHE_ROOT = Path("./cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Data and model paths
# -----------------------------
IMAGES_BASE_DIR = Path("data/vsr/demo_images")
PREDICTIONS_FILENAME = "predictions.json"
DEMO_INDICES_RANDOM = {643, 1030, 655, 1070, 687, 818, 53, 445, 960, 961, 1090, 452, 710, 199, 968, 213, 731, 606, 112, 368, 627, 885, 375, 1017}
DEMO_INDICES_ZEROSHOT = {10, 15, 144, 17, 21, 278, 23, 25, 28, 293, 173, 54, 62, 64, 70, 198, 199, 202, 91, 224, 98, 102, 104, 109, 241, 251}
VEA_BEST_LAYERS_IDXES_RANDOM = [17, 20, 24]
VEA_BEST_LAYERS_IDXES_ZEROSHOT = [17, 20, 10]

MODEL_PATHS = {
    "random": "modeling/Qwen3-VL-2B-Instruct_causal-lm_freeze-vision_vsr_b4x16_2e-05_e3_random-train-7680/checkpoint-260",
    "zeroshot": "modeling/Qwen3-VL-2B-Instruct_causal-lm_freeze-vision_vsr_b8x8_2e-05_e3_zeroshot-train-3489/checkpoint-160",
}
MODEL_REPOS = {
    "random": "changdb/Qwen3-VL-2B-Instruct-VSR-Random",
    "zeroshot": "changdb/Qwen3-VL-2B-Instruct-VSR-Zeroshot",
}

HEATMAP_DIRS = {
    "random": {
        "standard": f"{MODEL_PATHS["random"]}/results/random/dev/method-standard/visualizations/demo_heatmaps",
        "adaptvis": f"{MODEL_PATHS["random"]}/results/random/dev/method-adaptvis_threshold-0.8_sharpen-weight-1.2_smoothen-weight-0.2/visualizations/demo_heatmaps",
        "vea": f"{MODEL_PATHS["random"]}/results/random/dev/method-vea_smooth-strength-0.5_highlight-strength-0.5/visualizations/demo_heatmaps",
        "clvs": f"{MODEL_PATHS["random"]}/results/random/dev/method-clvs_smoothing-0.8_window-memory-size-0.8_uncertainty-threshold-0.5/visualizations/demo_heatmaps",
    },
    "zeroshot": {
        "standard": f"{MODEL_PATHS["zeroshot"]}/results/zeroshot/dev/method-standard/visualizations/demo_heatmaps",
        "adaptvis": f"{MODEL_PATHS["zeroshot"]}/results/zeroshot/dev/method-adaptvis_threshold-0.8_sharpen-weight-1.2_smoothen-weight-0.1/visualizations/demo_heatmaps",
        "vea": f"{MODEL_PATHS["zeroshot"]}/results/zeroshot/dev/method-vea_smooth-strength-0.8_highlight-strength-0.2/visualizations/demo_heatmaps",
        "clvs": f"{MODEL_PATHS["zeroshot"]}/results/zeroshot/dev/method-clvs_smoothing-0.7_window-memory-size-0.8_uncertainty-threshold-0.5/visualizations/demo_heatmaps",
    },
}

INTERVENTION_METHOD_DISPLAY = {
    "Adaptvis": "adaptvis",
    "Vea": "vea",
    "Clvs": "clvs",
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# -----------------------------
# 2) Hyperparameter specs
# -----------------------------
PARAM_SPECS = {
    "adaptvis": [
        {
            "name": "threshold",
            "label": "Threshold",
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
        },
        {
            "name": "sharpen_weight",
            "label": "Sharpen weight",
            "min": 1.0,
            "max": 2.0,
            "step": 0.1,
        },
        {
            "name": "smoothen_weight",
            "label": "Smoothen weight",
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
        },
    ],
    "vea": [
        {
            "name": "smooth_strength",
            "label": "Smooth strength",
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
        },
        {
            "name": "highlight_strength",
            "label": "Highlight strength",
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
        },
    ],
    "clvs": [
        {
            "name": "smoothing",
            "label": "Smoothing",
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
        },
        {
            "name": "window_memory_size",
            "label": "Window memory size",
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
        },
        {
            "name": "uncertainty_threshold",
            "label": "Uncertainty threshold",
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
        },
    ],
}

# -----------------------------
# 3) Caching
# -----------------------------

# Prevent duplicate work if two requests generate the same key simultaneously
_generate_lock = threading.Lock()
_model_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_processor():
    return AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")


@lru_cache(maxsize=2)
def get_model(split: str):
    if split not in MODEL_REPOS:
        raise ValueError(f"Unknown split: {split}")

    # Thread-safe guard against double-loading under concurrent requests
    with _model_lock:
        repo_id = MODEL_REPOS[split]
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            repo_id,
            attn_implementation="eager",
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        return model


def _canonicalize_params(
    params: dict[str, float], ndigits: int = 6
) -> dict[str, float]:
    """
    Make params stable so 0.3000000004 doesn't create a new cache entry.
    """
    return {k: round(float(params[k]), ndigits) for k in sorted(params.keys())}


def _cache_hash(
    split: str, method: str, image_key: str, example_idx: int, params: dict[str, float]
) -> str:
    payload = {
        "split": split,
        "method": method,
        "image_key": image_key,
        "example_idx": example_idx,
        "params": _canonicalize_params(params),
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def cache_path_for(
    split: str, method: str, image_key: str, example_idx: int, params: dict[str, float]
) -> Path:
    h = _cache_hash(split, method, image_key, example_idx, params)
    safe_key = image_key.replace("/", "__")  # safe filename
    out_dir = CACHE_ROOT / split / method
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir / f"{safe_key}__{h}.jpg"


def cache_paths_for(split: str, method: str, image_key: str, example_idx: int, params: dict[str, float]):
    img_path = cache_path_for(split, method, image_key, example_idx, params)
    meta_path = img_path.with_suffix(".json")
    return img_path, meta_path


def load_cached_result(img_path: Path, meta_path: Path):
    if not (img_path.exists() and meta_path.exists()):
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return str(img_path), meta["decoded_output"], float(meta["confidence"])


def save_cached_result(
    img_path: Path, 
    meta_path: Path, 
    original_img, 
    overlay_img, 
    decoded_output: str, 
    confidence: float, 
    meta_extra: dict | None = None
):
    meta = {
        "decoded_output": decoded_output,
        "confidence": float(confidence),
    }
    if meta_extra:
        meta.update(meta_extra)

    tmp_img = img_path.with_suffix(".tmp.jpg")
    tmp_meta = meta_path.with_suffix(".tmp.json")

    save_attention_image_overlay(
        img=original_img,
        attn_heatmap=overlay_img,
        output_file_path=tmp_img,
    )
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    tmp_img.replace(img_path)
    tmp_meta.replace(meta_path)


# -----------------------------
# 4) Utilities: indexing + defaults parsing
# -----------------------------
def _get_valid_prediction_indices(split: str) -> set[int]:
    return DEMO_INDICES_RANDOM if split == "random" else DEMO_INDICES_ZEROSHOT


def _resolve_dir(p: str) -> Path:
    return Path(p).expanduser().resolve()


def _collect_images(root: Path):
    # recursive glob, robust to nested dirs
    return [
        p for p in root.glob("**/*") if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]


def _key_for_file(root: Path, p: Path) -> str:
    # match by relative path WITHOUT extension (works even if extensions differ across methods)
    return p.relative_to(root).with_suffix("").as_posix()


def index_heatmap_dir(root: Path) -> dict[str, str]:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Heatmap directory not found: {root}")
    m = {}
    for p in _collect_images(root):
        k = _key_for_file(root, p)
        # keep first occurrence if duplicates
        if k not in m:
            m[k] = str(p)
    return m


def load_predictions_data(root: Path, split: str):
    with open(root / PREDICTIONS_FILENAME) as f:
        predictions = json.load(f)

    valid_indices = _get_valid_prediction_indices(split)
    predictions = [pred for i, pred in enumerate(predictions["prediction_results"]) if i in valid_indices]

    return predictions


def _extract_method_segment(path_str: str) -> str:
    # Finds the path segment like: method-vea_smooth-strength-0.8_highlight-strength-0.2
    parts = Path(path_str).parts
    for part in parts:
        if part.startswith("method-"):
            return part
    raise ValueError(f"Could not find 'method-*' segment in path: {path_str}")


def parse_default_params_from_path(path_str: str, method: str) -> dict[str, float]:
    """
    Parses defaults from directory name.
    Example: method-vea_smooth-strength-0.8_highlight-strength-0.2
    -> {"smooth_strength": 0.8, "highlight_strength": 0.2}
    """
    seg = _extract_method_segment(path_str)
    if seg == "method-standard":
        return {}

    prefix = f"method-{method}_"
    if not seg.startswith(prefix):
        # still try a best-effort parse
        # e.g., method might be "vea" and segment starts with method-vea_...
        pass

    # everything after 'method-<method>_' are underscore-separated tokens
    try:
        after = seg.split(f"method-{method}_", 1)[1]
    except Exception:
        # fallback: split after first underscore
        after = seg.split("_", 1)[1] if "_" in seg else ""

    defaults: dict[str, float] = {}
    if not after:
        return defaults

    tokens = after.split("_")
    for tok in tokens:
        # token format: <param-name-with-hyphens>-<float>
        parts = tok.split("-")
        if len(parts) < 2:
            continue
        raw_val = parts[-1]
        raw_key = "-".join(parts[:-1])
        try:
            val = float(raw_val)
        except ValueError:
            continue
        key = raw_key.replace("-", "_")
        defaults[key] = val

    return defaults


def get_default_params(split: str, method: str) -> dict[str, float]:
    return parse_default_params_from_path(HEATMAP_DIRS[split][method], method)


def params_close(a: dict[str, float], b: dict[str, float], tol: float = 1e-9) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if abs(float(a[k]) - float(b[k])) > tol:
            return False
    return True


@lru_cache(maxsize=None)
def load_split_index(split: str):
    """
    Loads and caches all 4 method maps for a split:
      maps[method][key] -> filepath
    Keys list is derived from standard as canonical ordering.
    """
    if split not in HEATMAP_DIRS:
        raise ValueError(f"Unknown split: {split}")

    maps = {}
    predictions = {}
    missing_dirs = []
    for method, p in HEATMAP_DIRS[split].items():
        root = _resolve_dir(p)
        if not root.exists():
            missing_dirs.append(str(root))
            continue

        maps[method] = index_heatmap_dir(root)
        predictions[method] = load_predictions_data(Path(p).parent.parent, split)

    if missing_dirs:
        raise FileNotFoundError(
            "Some heatmap directories were not found.\n"
            + "\n".join(f"- {d}" for d in missing_dirs)
        )

    std = maps["standard"]
    keys = sorted(std.keys())

    status = f"Indexed split **{split}**:\n" f"- **{len(keys)}** images\n- **{len(predictions[method])}** examples"

    return {"maps": maps, "predictions": predictions, "keys": keys, "status": status}


# -----------------------------
# 5) Run model inference
# -----------------------------
def generate_model_outputs(
    model: Qwen3VLForConditionalGeneration,
    encodings: dict[str, torch.Tensor],
    weight=None,
    use_clvs=False,
    smoothing=None,
    window_memory_size=None,
    uncertainty_threshold=None,
):
    image_token_indices = (
        encodings["input_ids"][0] == model.config.image_token_id
    ).nonzero(as_tuple=True)[0]

    with torch.inference_mode():
        model_outputs = model.generate(
            **encodings,
            image_token_indices=image_token_indices,
            weight=weight,
            max_new_tokens=1,
            use_cache=True,
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            use_clvs=use_clvs,
            smoothing=smoothing,
            window_memory_size=window_memory_size,
            uncertainty_threshold=uncertainty_threshold,
        )

    return model_outputs


def get_adaptvis_outputs(
    model: Qwen3VLForConditionalGeneration,
    encodings: dict[str, torch.Tensor],
    standard_method_confidence: float,
    params: dict[str, float],
):
    if standard_method_confidence < params["threshold"]:
        weight = params["smoothen_weight"]
    else:
        weight = params["sharpen_weight"]

    model_outputs = generate_model_outputs(model, encodings, weight)
    return model_outputs


def get_vea_outputs(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    encodings: dict[str, torch.Tensor],
    best_layers_idxes: list[int],
    params: dict[str, float],
):
    patch_size = model.config.vision_config.patch_size
    spatial_merge_size = model.config.vision_config.spatial_merge_size
    temporal_patch_size = model.config.vision_config.temporal_patch_size

    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    _, num_patches_h, num_patches_w = [
        grid_dim.item() for grid_dim in encodings["image_grid_thw"][0]
    ]
    image_h = num_patches_h * patch_size
    image_w = num_patches_w * patch_size

    model_outputs = generate_model_outputs(model, encodings)

    first_output_token_idx = 0

    # Attention Extraction
    evidence_scores = sum(
        [
            model_outputs.attentions[first_output_token_idx][
                layer_idx
            ].image_attention_weights
            for layer_idx in best_layers_idxes
        ]
    ) / len(best_layers_idxes)
    evidence_scores = torch.mean(evidence_scores, dim=1)
    evidence_scores = evidence_scores.reshape(
        num_patches_h // spatial_merge_size, num_patches_w // spatial_merge_size
    )

    # Attention Mask Denoising
    e_denoised = denoise_attention_mask(
        evidence_scores.numpy(),
        (num_patches_h // spatial_merge_size, num_patches_w // spatial_merge_size),
    )

    # Highlight Mask Smoothing
    e_smoothed = smooth_highlight_mask(
        e_denoised,
        grid_hw=(
            num_patches_h // spatial_merge_size,
            num_patches_w // spatial_merge_size,
        ),
        image_short_side=min(image_h, image_w),
        patch_size=patch_size,
        smooth_strength=params["smooth_strength"],
        return_grid=True,
    )
    patches_smoothed = e_smoothed.repeat(spatial_merge_size, axis=0).repeat(
        spatial_merge_size, axis=1
    )
    patches_smoothed = (
        patches_smoothed.reshape(
            num_patches_h // spatial_merge_size,
            spatial_merge_size,
            num_patches_w // spatial_merge_size,
            spatial_merge_size,
        )
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )

    scale = (
        params["highlight_strength"]
        + (1.0 - params["highlight_strength"]) * patches_smoothed
    )  # (H,W)
    scale = scale.reshape(-1, 1)

    pixel_values = encodings["pixel_values"]
    patch_dim = int(pixel_values.shape[1])

    c = len(image_mean)
    denom = c * patch_size * patch_size
    if patch_dim % denom != 0:
        raise ValueError(
            f"patch_dim={patch_dim} not divisible by (3 * patch_size * patch_size)={denom}"
        )

    pixel_values = pixel_values.view(
        num_patches_h * num_patches_w, temporal_patch_size, c, patch_size, patch_size
    )

    mean = torch.tensor(
        image_mean, device=pixel_values.device, dtype=pixel_values.dtype
    ).view(1, 1, c, 1, 1)
    std = torch.tensor(
        image_std, device=pixel_values.device, dtype=pixel_values.dtype
    ).view(1, 1, c, 1, 1)

    # denormalize and apply scale to pixel_values
    denormalized_pixel_values = pixel_values * std + mean
    denormalized_pixel_values_aug = denormalized_pixel_values * scale.reshape(
        -1, 1, 1, 1, 1
    )

    # re-normalize
    pixel_values_aug = (denormalized_pixel_values_aug - mean) / std
    pixel_values_aug = pixel_values_aug.reshape(
        num_patches_h * num_patches_w, patch_dim
    )
    encodings["pixel_values"] = pixel_values_aug

    model_outputs = generate_model_outputs(model, encodings)

    return model_outputs


def get_clvs_outputs(
    model: Qwen3VLForConditionalGeneration,
    encodings: dict[str, torch.Tensor],
    params: dict[str, float],
):
    use_clvs = True
    smoothing = params["smoothing"]
    window_memory_size = params["window_memory_size"]
    uncertainty_threshold = params["uncertainty_threshold"]

    model_outputs = generate_model_outputs(
        model, 
        encodings, 
        weight=None, 
        use_clvs=use_clvs, 
        smoothing=smoothing, 
        window_memory_size=window_memory_size, 
        uncertainty_threshold=uncertainty_threshold,
    )
    return model_outputs


def generate_custom_heatmap_overlay(
    *,
    split: str,
    method: str,
    image_key: str,
    example_idx: int,
    model_prompt: str,
    params: dict[str, float],
    standard_method_confidence: float,
):
    # 1) Disk cache check
    img_path, meta_path = cache_paths_for(split, method, image_key, example_idx, params)

    cached = load_cached_result(img_path, meta_path)
    if cached is not None:
        return cached

    with _generate_lock:
        # Re-check after acquiring lock (double-checked locking)
        cached = load_cached_result(img_path, meta_path)
        if cached is not None:
            return cached

        model = get_model(split)
        processor = get_processor()

        patch_size = model.config.vision_config.patch_size
        spatial_merge_size = model.config.vision_config.spatial_merge_size

        prompt_text = processor.apply_chat_template(
            model_prompt, tokenize=False, add_generation_prompt=True
        )
        image = cv2.imread(f"{IMAGES_BASE_DIR}/{image_key}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        processor.tokenizer.padding_side = "left"
        encodings = processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            return_token_type_ids=False,
            device="cpu",
        )
        _, num_patches_h, num_patches_w = [
            grid_dim.item() for grid_dim in encodings["image_grid_thw"][0]
        ]

        if method == "adaptvis":
            model_outputs = get_adaptvis_outputs(
                model, encodings, standard_method_confidence, params
            )
        elif method == "vea":
            best_layers_idxes = VEA_BEST_LAYERS_IDXES_RANDOM if split == "random" else VEA_BEST_LAYERS_IDXES_ZEROSHOT
            model_outputs = get_vea_outputs(model, processor, encodings, best_layers_idxes, params)
        elif method == "clvs":
            model_outputs = get_clvs_outputs(model, encodings, params)
        else:
            raise ValueError("Unknown method. Valid methods: [adaptvis, vea, clvs]")

        first_output_token_idx = 0
        final_layer_image_attention_weights = model_outputs.attentions[
            first_output_token_idx
        ][-1].image_attention_weights
        image_attn_weights_avg_heads = torch.mean(
            final_layer_image_attention_weights, dim=1
        )
        image_attn_weights_avg_heads = image_attn_weights_avg_heads.squeeze().reshape(
            num_patches_h // spatial_merge_size, num_patches_w // spatial_merge_size
        )

        # rescale attention weights to the resized image from the processor
        rescaled_avg_attn_weights = F.interpolate(
            image_attn_weights_avg_heads.unsqueeze(0).unsqueeze(0),
            scale_factor=patch_size * spatial_merge_size,
            mode="bicubic",
        )[0]
        rescaled_avg_attn_weights = rescaled_avg_attn_weights.permute(1, 2, 0)

        # resize original image to the resized image from the processor
        image = cv2.resize(
            image,
            (num_patches_w * patch_size, num_patches_h * patch_size),
            interpolation=cv2.INTER_CUBIC,
        )

        prompt_len = encodings["input_ids"].shape[1]
        new_tokens = model_outputs.sequences[first_output_token_idx][prompt_len:]
        decoded_outputs = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        decoded_output = decoded_outputs[0]

        confidence = np.round(float(max(torch.softmax(model_outputs.scores[first_output_token_idx].squeeze(), dim=0))), 2)
        save_cached_result(
            img_path=img_path,
            meta_path=meta_path,
            original_img=image,
            overlay_img=rescaled_avg_attn_weights,
            decoded_output=decoded_output,
            confidence=confidence,
            meta_extra={
                "split": split,
                "method": method,
                "image_key": image_key,
                "params": _canonicalize_params(params),
            }
        )

        return str(img_path), decoded_output, confidence


# -----------------------------
# 6) UI logic
# -----------------------------
def warmup(split: str, intervention_method_display: str):
    # Load default split model at startup
    _ = get_model("random")
    _ = get_processor()

    return on_context_change(split, intervention_method_display)


def slider_updates_for(split: str, method: str):
    """
    Returns gr.update() for up to 3 sliders + markdown describing defaults.
    """
    specs = PARAM_SPECS[method]
    defaults = get_default_params(split, method)

    # Build a pretty defaults line
    pretty = []
    for s in specs:
        pretty.append(f"`{s['name']}`={defaults.get(s['name'], None)}")
    defaults_md = "Default hyperparameters: " + ", ".join(pretty)

    # We use 3 sliders max; hide unused.
    updates = []
    for i in range(3):
        if i < len(specs):
            s = specs[i]
            updates.append(
                gr.update(
                    visible=True,
                    interactive=True,
                    label=s["label"],
                    minimum=s["min"],
                    maximum=s["max"],
                    step=s["step"],
                    value=float(defaults[s["name"]]),
                )
            )
        else:
            updates.append(
                gr.update(
                    visible=False,
                    interactive=False,  # hidden + disabled
                )
            )

    return updates[0], updates[1], updates[2], defaults_md


def collect_params(method: str, p1: float, p2: float, p3: float) -> dict[str, float]:
    specs = PARAM_SPECS[method]
    vals = [p1, p2, p3]
    out = {}
    for i, s in enumerate(specs):
        out[s["name"]] = float(vals[i])
    return out


def clamp_idx(idx: int, n: int) -> int:
    if n <= 0:
        return 0
    return max(0, min(int(idx), n - 1))


def render_view_only(
    split: str, intervention_method_display: str, idx: int, p1: float, p2: float, p3: float
):
    """
    Show precomputed intervention method heatmap if params == defaults.
    Otherwise shows None (pending) and enables the Generate button
    """
    intervention_method = INTERVENTION_METHOD_DISPLAY[intervention_method_display]
    data = load_split_index(split)
    keys = data["keys"]
    maps = data["maps"]
    predictions = data["predictions"]

    num_examples = len(predictions["standard"])

    if not keys:
        return (
            None, None,
            "", "",
            "", 0.0,
            "", 0.0,
            "**0 / 0**",
            "", 0,
            gr.update(visible=False),
            data["status"],
            "No images found.",
            gr.update(interactive=False),
        )

    idx = clamp_idx(idx, num_examples)

    prediction_result = predictions["standard"][idx]
    image_filename_key = prediction_result["image_filename"].replace(".jpg", "")
    prompt = prediction_result["prompt"][0]["content"][-1]["text"]
    label = prediction_result["label"]
    standard_prediction = prediction_result["prediction"]
    standard_confidence = prediction_result["confidence"]

    standard_method_path = maps["standard"][image_filename_key]
    chosen_params = collect_params(intervention_method, p1, p2, p3)
    default_params = get_default_params(split, intervention_method)

    is_default = params_close(chosen_params, default_params)
    gen_btn_update = gr.update(interactive=(not is_default))

    method_prediction = ""
    method_confidence = 0.0
    if is_default:
        prediction_result = predictions[intervention_method][idx]
        image_filename_key = prediction_result["image_filename"].replace(".jpg", "")
        method_prediction = prediction_result["prediction"]
        method_confidence = prediction_result["confidence"]

        intervention_method_path = maps[intervention_method].get(image_filename_key)
        if intervention_method_path is None:
            intervention_method_path_img = None
            run_status = f"⚠️ Missing `{intervention_method}` heatmap for key: `{image_filename_key}`"
        else:
            intervention_method_path_img = intervention_method_path
            run_status = "✅ Showing heatmap with default hyperparameters. Generate button disabled."
    else:
        intervention_method_path_img = None
        run_status = "🟡 Custom hyperparameters selected. Click **Generate Heatmap** to run inference."

    counter = f"**{idx + 1} / {num_examples}**"
    idx_slider_update = gr.update(
        visible=True, minimum=0, maximum=num_examples - 1, value=idx
    )

    return (
        standard_method_path,
        intervention_method_path_img,
        prompt,
        label,
        standard_prediction,
        standard_confidence,
        method_prediction,
        method_confidence,
        counter,
        image_filename_key,
        idx,
        idx_slider_update,
        data["status"],
        run_status,
        gen_btn_update,
    )


def render_with_generation(
    split: str, intervention_method_display: str, idx: int, p1: float, p2: float, p3: float
):
    """
    Called only by the Generate Heatmap button click.
    """
    intervention_method = INTERVENTION_METHOD_DISPLAY[intervention_method_display]
    data = load_split_index(split)
    keys = data["keys"]
    maps = data["maps"]
    predictions = data["predictions"]

    num_examples = len(predictions["standard"])

    if not keys:
        return (
            None, None,
            "", "",
            "", 0.0,
            "", 0.0,
            "**0 / 0**",
            "", 0,
            gr.update(visible=False),
            data["status"],
            "No images found.",
            gr.update(interactive=False),
        )

    idx = clamp_idx(idx, num_examples)

    prediction_result = predictions["standard"][idx]
    image_filename_key = prediction_result["image_filename"].replace(".jpg", "")
    model_prompt = prediction_result["prompt"]
    prompt = model_prompt[0]["content"][-1]["text"]
    label = prediction_result["label"]
    standard_prediction = prediction_result["prediction"]
    standard_confidence = prediction_result["confidence"]

    standard_method_path = maps["standard"][image_filename_key]
    chosen_params = collect_params(intervention_method, p1, p2, p3)
    default_params = get_default_params(split, intervention_method)

    # Button should be disabled in this case, but handle anyway
    if params_close(chosen_params, default_params):
        intervention_method_path = maps[intervention_method].get(image_filename_key)
        intervention_method_path_img = intervention_method_path if intervention_method_path is not None else None
        run_status = "Defaults selected — nothing to generate. (Showing precomputed if available.)"
        gen_btn_update = gr.update(interactive=False)
    else:
        intervention_method_path_img, method_prediction, method_confidence = generate_custom_heatmap_overlay(
            split=split,
            method=intervention_method,
            image_key=image_filename_key,
            example_idx=idx,
            model_prompt=model_prompt,
            params=chosen_params,
            standard_method_confidence=standard_confidence,
        )
        run_status = "✅ Generated heatmap with custom hyperparameters."
        gen_btn_update = gr.update(interactive=True)

    counter = f"**{idx + 1} / {num_examples}**"
    idx_slider_update = gr.update(
        visible=True, minimum=0, maximum=num_examples - 1, value=idx
    )

    return (
        standard_method_path,
        intervention_method_path_img,
        prompt,
        label,
        standard_prediction,
        standard_confidence,
        method_prediction,
        method_confidence,
        counter,
        image_filename_key,
        idx,
        idx_slider_update,
        data["status"],
        run_status,
        gen_btn_update,
    )


def on_context_change(split: str, intervention_method_display: str):
    """
    Split/method changed:
      - reset idx to 0
      - reset sliders to defaults
      - show precomputed
      - disable Generate button
    """
    intervention_method = INTERVENTION_METHOD_DISPLAY[intervention_method_display]
    try:
        u1, u2, u3, defaults_md = slider_updates_for(split, intervention_method)

        defaults = get_default_params(split, intervention_method)
        specs = PARAM_SPECS[intervention_method]
        v1 = float(defaults[specs[0]["name"]])
        v2 = float(defaults[specs[1]["name"]]) if len(specs) > 1 else 0.0
        v3 = float(defaults[specs[2]["name"]]) if len(specs) > 2 else 0.0

        (
            standard_method_path,
            intervention_method_path_img,
            caption,
            label,
            standard_prediction,
            standard_confidence,
            method_prediction,
            method_confidence,
            counter,
            key,
            idx,
            idx_slider_upd,
            index_status,
            run_status,
            gen_btn_upd,
        ) = render_view_only(split, intervention_method_display, 0, v1, v2, v3)

        return (
            standard_method_path,
            intervention_method_path_img,
            caption,
            label,
            standard_prediction,
            standard_confidence,
            method_prediction,
            method_confidence,
            counter,
            key,
            idx,
            idx_slider_upd,
            defaults_md,
            index_status,
            run_status,
            gen_btn_upd,
            u1,
            u2,
            u3,
        )
    except Exception as e:
        error_msg = traceback.format_exc()
        err = f"❌ {type(e).__name__}: {error_msg}"

        return (
            None, None,
            "", "",
            "", 0.0,
            "", 0.0,
            "**0 / 0**",
            "", 0,
            gr.update(visible=False),
            "Defaults unavailable.",
            err, "",
            gr.update(interactive=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def on_prev(split, intervention_method_display, idx, p1, p2, p3):
    return render_view_only(split, intervention_method_display, int(idx) - 1, p1, p2, p3)


def on_next(split, intervention_method_display, idx, p1, p2, p3):
    return render_view_only(split, intervention_method_display, int(idx) + 1, p1, p2, p3)


def on_random(split, intervention_method_display, idx, p1, p2, p3):
    data = load_split_index(split)
    valid_indices = _get_valid_prediction_indices(split)
    n = len(valid_indices)

    if n == 0:
        return (
            None, None,
            "", "",
            "", 0.0,
            "", 0.0,
            "**0 / 0**",
            "", 0,
            gr.update(visible=False),
            data["status"],
            "No images found.",
            gr.update(interactive=False),
        )
    ridx = random.randrange(n)
    return render_view_only(split, intervention_method_display, ridx, p1, p2, p3)


# -----------------------------
# 7) Build Gradio UI
# -----------------------------
with gr.Blocks(title="Visual Attention Intervention for VSR") as demo:
    gr.Markdown("# Survey of Visual Attention Intervention methods\n## Comparing Visual Heatmaps from Qwen-VL 2B on VSR dataset")

    with gr.Row():
        with gr.Column(scale=1):
            split_dd = gr.Radio(
                choices=["random", "zeroshot"], value="random", label="Dataset split"
            )

        with gr.Column(scale=7):
            pass

    with gr.Row():
        with gr.Column():
            with gr.Row():
                prev_btn = gr.Button("⬅️ Prev")
                next_btn = gr.Button("Next ➡️")
                rand_btn = gr.Button("🎲 Random")
                counter_md = gr.Markdown("**0 / 0**")

        with gr.Column():
            idx_state = gr.State(0)
            idx_slider = gr.Slider(
                minimum=0, maximum=0, value=0, step=1, label="Index", visible=True
            )
            key_box = gr.Textbox(label="Current key", interactive=False, visible=False)

    index_status_md = gr.Markdown()
    run_status_md = gr.Markdown()

    gr.Markdown("---")

    with gr.Row():
        caption = gr.Textbox(label="Prompt", interactive=False)
        label = gr.Textbox(label="Label", interactive=False)

    with gr.Row():
        with gr.Column():
            pass
        with gr.Column():
            intervention_method_dd = gr.Radio(
                choices=list(INTERVENTION_METHOD_DISPLAY.keys()),
                value="Adaptvis",
                label="Intervention Method",
            )
            defaults_md = gr.Markdown()
            # Hyperparameter controls (3 sliders max; we re-label + show/hide dynamically)
            with gr.Row():
                p1 = gr.Slider(
                    label="param1",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    visible=True,
                    interactive=True,
                )
                p2 = gr.Slider(
                    label="param2",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    visible=True,
                    interactive=True,
                )
                p3 = gr.Slider(
                    label="param3",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    visible=True,
                    interactive=True,
                )

            with gr.Row():
                gen_btn = gr.Button(
                    "Generate Heatmap", variant="secondary", interactive=False
                )

    with gr.Row():
        with gr.Column():
            with gr.Row():
                standard_prediction = gr.Textbox(label="Prediction", interactive=False)
                standard_confidence = gr.Textbox(label="Confidence", interactive=False)
            img_standard = gr.Image(label="Qwen3-VL 2B", interactive=False)
        with gr.Column():
            with gr.Row():
                method_prediction = gr.Textbox(label="Prediction", interactive=False)
                method_confidence = gr.Textbox(label="Confidence", interactive=False)
            img_intervention = gr.Image(label="Qwen3-VL 2B + Intervention Method", interactive=False)

    # Split / method change => reset sliders to defaults + show precomputed + disable button
    split_dd.change(
        fn=on_context_change,
        inputs=[split_dd, intervention_method_dd],
        outputs=[
            img_standard,
            img_intervention,
            caption,
            label,
            standard_prediction,
            standard_confidence,
            method_prediction,
            method_confidence,
            counter_md,
            key_box,
            idx_state,
            idx_slider,
            defaults_md,
            index_status_md,
            run_status_md,
            gen_btn,
            p1,
            p2,
            p3,
        ],
        show_progress="hidden",
    )
    intervention_method_dd.change(
        fn=on_context_change,
        inputs=[split_dd, intervention_method_dd],
        outputs=[
            img_standard,
            img_intervention,
            caption,
            label,
            standard_prediction,
            standard_confidence,
            method_prediction,
            method_confidence,
            counter_md,
            key_box,
            idx_state,
            idx_slider,
            defaults_md,
            index_status_md,
            run_status_md,
            gen_btn,
            p1,
            p2,
            p3,
        ],
        show_progress="hidden",
    )

    # Navigation buttons (keep current hyperparams)
    prev_btn.click(
        fn=on_prev,
        inputs=[split_dd, intervention_method_dd, idx_state, p1, p2, p3],
        outputs=[
            img_standard,
            img_intervention,
            caption,
            label,
            standard_prediction,
            standard_confidence,
            method_prediction,
            method_confidence,
            counter_md,
            key_box,
            idx_state,
            idx_slider,
            index_status_md,
            run_status_md,
            gen_btn,
        ],
        show_progress="hidden",
    )
    next_btn.click(
        fn=on_next,
        inputs=[split_dd, intervention_method_dd, idx_state, p1, p2, p3],
        outputs=[
            img_standard,
            img_intervention,
            caption,
            label,
            standard_prediction,
            standard_confidence,
            method_prediction,
            method_confidence,
            counter_md,
            key_box,
            idx_state,
            idx_slider,
            index_status_md,
            run_status_md,
            gen_btn,
        ],
        show_progress="hidden",
    )
    rand_btn.click(
        fn=on_random,
        inputs=[split_dd, intervention_method_dd, idx_state, p1, p2, p3],
        outputs=[
            img_standard,
            img_intervention,
            caption,
            label,
            standard_prediction,
            standard_confidence,
            method_prediction,
            method_confidence,
            counter_md,
            key_box,
            idx_state,
            idx_slider,
            index_status_md,
            run_status_md,
            gen_btn,
        ],
        show_progress="hidden",
    )

    # Jump by slider
    idx_slider.change(
        fn=render_view_only,
        inputs=[split_dd, intervention_method_dd, idx_slider, p1, p2, p3],
        outputs=[
            img_standard,
            img_intervention,
            caption,
            label,
            standard_prediction,
            standard_confidence,
            method_prediction,
            method_confidence,
            counter_md,
            key_box,
            idx_state,
            idx_slider,
            index_status_md,
            run_status_md,
            gen_btn,
        ],
        show_progress="hidden",
    )

    # Hyperparameter edits used for re-render of heatmaps
    p1.release(
        fn=render_view_only,
        inputs=[split_dd, intervention_method_dd, idx_state, p1, p2, p3],
        outputs=[
            img_standard,
            img_intervention,
            caption,
            label,
            standard_prediction,
            standard_confidence,
            method_prediction,
            method_confidence,
            counter_md,
            key_box,
            idx_state,
            idx_slider,
            index_status_md,
            run_status_md,
            gen_btn,
        ],
        show_progress="hidden",
    )
    p2.release(
        fn=render_view_only,
        inputs=[split_dd, intervention_method_dd, idx_state, p1, p2, p3],
        outputs=[
            img_standard,
            img_intervention,
            caption,
            label,
            standard_prediction,
            standard_confidence,
            method_prediction,
            method_confidence,
            counter_md,
            key_box,
            idx_state,
            idx_slider,
            index_status_md,
            run_status_md,
            gen_btn,
        ],
        show_progress="hidden",
    )
    p3.release(
        fn=render_view_only,
        inputs=[split_dd, intervention_method_dd, idx_state, p1, p2, p3],
        outputs=[
            img_standard,
            img_intervention,
            caption,
            label,
            standard_prediction,
            standard_confidence,
            method_prediction,
            method_confidence,
            counter_md,
            key_box,
            idx_state,
            idx_slider,
            index_status_md,
            run_status_md,
            gen_btn,
        ],
        show_progress="hidden",
    )

    # Generate button => run model inference to generate heatmaps
    gen_btn.click(
        fn=render_with_generation,
        inputs=[split_dd, intervention_method_dd, idx_state, p1, p2, p3],
        outputs=[
            img_standard,
            img_intervention,
            caption,
            label,
            standard_prediction,
            standard_confidence,
            method_prediction,
            method_confidence,
            counter_md,
            key_box,
            idx_state,
            idx_slider,
            index_status_md,
            run_status_md,
            gen_btn,
        ],
        show_progress="full",
        show_progress_on=img_intervention,
    )

    # Initialize on app load
    demo.load(
        fn=warmup,
        inputs=[split_dd, intervention_method_dd],
        outputs=[
            img_standard,
            img_intervention,
            caption,
            label,
            standard_prediction,
            standard_confidence,
            method_prediction,
            method_confidence,
            counter_md,
            key_box,
            idx_state,
            idx_slider,
            defaults_md,
            index_status_md,
            run_status_md,
            gen_btn,
            p1,
            p2,
            p3,
        ],
    )

if __name__ == "__main__":
    demo.queue().launch()
