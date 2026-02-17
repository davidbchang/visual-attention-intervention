import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from aim import Run
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from visual_attention_intervention.base import (
    VEAConfig,
    get_experiment_output_paths, 
    MODELS_BASE_DIR, 
    PRETRAINED_MODELS_BASE_DIR
)
from visual_attention_intervention.modeling_paligemma2 import PaliGemmaForConditionalGeneration
from visual_attention_intervention.utils import (
    compute_auroc,
    compute_patch_labels,
    denoise_attention_mask, 
    paligemma_data_collator,
    plot_layer_attentions,
    save_attention_image_overlay,
    smooth_highlight_mask,
    track_aim_run,
)
from visual_attention_intervention.vsr_dataset import VSRDataset


def get_best_layers(
    dataloader: DataLoader, 
    model: PaliGemmaForConditionalGeneration, 
    processor: AutoProcessor,
    coco_annotations_dir: str,
    device: torch.device,
    data_subset_size: int = 100,
    top_k: float = 0.1
):
    num_hidden_layers = model.config.text_config.num_hidden_layers
    patch_size = model.config.vision_config.patch_size
    preprocessed_image_size = processor.image_processor.size

    auroc_sum_per_layer = [0] * num_hidden_layers

    coco_data_val = COCO(f"{coco_annotations_dir}/instances_val2017.json")
    coco_data_train = COCO(f"{coco_annotations_dir}/instances_train2017.json")

    num_examples_processed = 0
    for input_data in tqdm(dataloader, total=data_subset_size):
        if num_examples_processed == data_subset_size:
            break

        captions, _, images, image_names, image_links, entities = input_data
        encodings = processor(text=captions, images=images, return_tensors="pt", padding="longest", device=device)

        if not entities:
            raise ValueError(f"No entities found for images: {image_names}.")

        entities = entities[0]
        image = images[0]
        image_name = image_names[0]
        image_link = image_links[0]

        h, w, _ = image.shape
        image_id = int(image_name.replace(".jpg", ""))

        # get bboxes from coco dataset
        coco_data = coco_data_train if "train" in image_link.split('/')[3] else coco_data_val
        cat_ids = coco_data.getCatIds(catNms=entities)
        annotations = coco_data.loadAnns(
            coco_data.getAnnIds(
                imgIds=image_id,
                catIds=cat_ids
            )
        )
        entity_bboxes = [
            [
                annot["bbox"][0], 
                annot["bbox"][1], 
                annot["bbox"][0] + annot["bbox"][2], 
                annot["bbox"][1] + annot["bbox"][3]
            ] 
            for annot in annotations
        ]

        patch_labels = compute_patch_labels(
            entity_bboxes, 
            image_size=(preprocessed_image_size["width"] * patch_size, preprocessed_image_size["height"] * patch_size), 
            patch_size=patch_size,
            bbox_source_size=(w, h)
        )
        if len(np.unique(np.asarray(patch_labels))) < 2:
            print(f"Only 1 entity found for image {image_name}, skipping example")
            continue

        image_token_indices = torch.where(encodings["input_ids"] == model.config.image_token_index, 1, 0)

        with torch.inference_mode():
            model_outputs = model.generate(
                **encodings, 
                image_token_indices=image_token_indices, 
                max_new_tokens=1, 
                use_cache=True, 
                output_attentions=True, 
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        first_output_token_idx = 0
        for j, layer_attentions in enumerate(model_outputs.attentions[first_output_token_idx]):
            image_attention_weights = layer_attentions.image_attention_weights
            image_attn_weights_avg_heads = torch.mean(image_attention_weights, dim=1)
            auroc = compute_auroc(image_attn_weights_avg_heads, patch_labels)
            auroc_sum_per_layer[j] += auroc

        num_examples_processed += 1

    avg_auroc_per_layer = [auroc_sum / data_subset_size for auroc_sum in auroc_sum_per_layer]
    sorted_avg_auroc_per_layer = sorted(enumerate(avg_auroc_per_layer), key=lambda x: x[1], reverse=True)
    best_layers_idxes = [layer_auroc[0] for layer_auroc in sorted_avg_auroc_per_layer[:math.ceil(len(sorted_avg_auroc_per_layer) * top_k)]]

    return best_layers_idxes


# def compute_auroc(
#     attention: np.ndarray,
#     labels: np.ndarray,
# ) -> float:
#     """
#     Compute AUROC(y_I, a_I) for a single image and layer.

#     Args:
#         attention: Visual attention vector of shape (256,).
#                    Higher means more attention to that patch.
#         labels: Binary label vector of shape (256,), values in {0,1}.

#     Returns:
#         AUROC score (float in [0, 1]).
#     """
#     attention = np.asarray(attention).reshape(-1)
#     labels = np.asarray(labels).reshape(-1)

#     assert attention.shape == labels.shape, f"Attention and labels must have same shape, {attention.shape} vs {labels.shape}"
#     assert set(np.unique(labels)).issubset({0, 1}), "Labels must be binary"

#     # AUROC is undefined if only one class is present
#     if len(np.unique(labels)) < 2:
#         return float("nan")

#     return roc_auc_score(labels, attention)


# def compute_patch_labels(
#     bboxes: list[tuple[float, float, float, float]],
#     image_size: tuple[int, int],
#     patch_size: int,
#     bbox_source_size: tuple[int, int] | None = None,  # (source_width, source_height)
#     clip: bool = True,
# ) -> np.ndarray:
#     """
#     Compute binary patch-level evidence labels y_I for an image in 224x224 patch space.

#     Args:
#         bboxes:
#             List of bounding boxes (x_min, y_min, x_max, y_max) in pixel coordinates
#             defined on the *source/original* image coordinate system.
#         image_size:
#             Target image size (default 224).
#         patch_size:
#             Patch size (default 14). For 224/14 -> 16x16=256 patches.
#         bbox_source_size:
#             (source_width, source_height) that the bbox coordinates correspond to.
#             - If provided: bboxes are scaled to the target (image_size, image_size).
#             - If None: bboxes are assumed already in target coordinates.
#         clip:
#             If True, clips scaled bboxes into [0, image_size].

#     Returns:
#         y_I: np.ndarray of shape (256,) with values in {0,1}, row-major order.
#     """
#     image_w, image_h = image_size
#     num_patches_h = image_h // patch_size
#     num_patches_w = image_w // patch_size
#     num_patches = num_patches_h * num_patches_w
#     patch_labels = np.zeros(num_patches, dtype=np.int64)

#     if not bboxes:
#         return patch_labels

#     # Scale factors from bbox-source coords -> target coords
#     if bbox_source_size is None:
#         scale_x = 1.0
#         scale_y = 1.0
#     else:
#         src_w, src_h = bbox_source_size
#         if src_w <= 0 or src_h <= 0:
#             raise ValueError(f"bbox_source_size must be positive, got {bbox_source_size}")
#         scale_x = image_w / float(src_w)
#         scale_y = image_h / float(src_h)

#     eps = 1e-6  # helps avoid including adjacent patches when a boundary falls exactly on a patch edge

#     for (x1, y1, x2, y2) in bboxes:
#         # Scale to target coordinate system
#         x1 *= scale_x
#         x2 *= scale_x
#         y1 *= scale_y
#         y2 *= scale_y

#         # Ensure proper ordering
#         x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
#         y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)

#         # Optionally clip to image bounds
#         if clip:
#             x_min = float(np.clip(x_min, 0.0, float(image_w)))
#             x_max = float(np.clip(x_max, 0.0, float(image_w)))
#             y_min = float(np.clip(y_min, 0.0, float(image_h)))
#             y_max = float(np.clip(y_max, 0.0, float(image_h)))

#         # Skip degenerate or empty boxes
#         if x_max <= x_min + eps or y_max <= y_min + eps:
#             continue

#         # Compute patch index ranges overlapped by the bbox
#         col_start = int(np.floor(x_min / patch_size))
#         col_end   = int(np.ceil((x_max - eps) / patch_size))
#         row_start = int(np.floor(y_min / patch_size))
#         row_end   = int(np.ceil((y_max - eps) / patch_size))

#         # Clip patch indices
#         col_start = max(col_start, 0)
#         row_start = max(row_start, 0)
#         col_end   = min(col_end, num_patches_w - 1)
#         row_end   = min(row_end, num_patches_h - 1)

#         if col_end < col_start or row_end < row_start:
#             continue

#         # Mark all overlapped patches as evidence
#         for r in range(row_start, row_end + 1):
#             start = r * num_patches_w + col_start
#             end = r * num_patches_w + col_end + 1
#             patch_labels[start:end] = 1

#     return patch_labels


def get_rescaled_avg_attn_weights(input_image_path, image_h, image_w, model_patch_size):
    avg_attn_weights = plt.imread(input_image_path)
    avg_attn_weights = np.transpose(avg_attn_weights, (2, 0, 1))
    rescaled_avg_attn_weights = F.interpolate(torch.from_numpy(avg_attn_weights.copy()).unsqueeze(0), scale_factor=model_patch_size, mode='bicubic')[0]
    rescaled_avg_attn_weights = F.interpolate(rescaled_avg_attn_weights.unsqueeze(0), (image_h, image_w), mode='bilinear')[0].cpu().numpy()   # (c, h, w)
    rescaled_avg_attn_weights = np.transpose(rescaled_avg_attn_weights, (1, 2, 0))    # (h, w, c)

    return rescaled_avg_attn_weights


# def save_attention_image_overlay(img, attn_heatmap, output_file_path, alpha=0.5):
#     fig, _ = plt.subplots()
#     plt.imshow(img)
#     plt.imshow(attn_heatmap, alpha=alpha)
#     plt.axis('off')
#     plt.savefig(output_file_path, bbox_inches="tight")
#     plt.close(fig)


# def plot_layer_attentions(output_layer_attentions, image_id, output_file_path):
#     layer_image_attention_weights = []
#     layer_text_attention_weights = []

#     for layer_attentions in output_layer_attentions:
#         image_attention_weights = layer_attentions.image_attention_weights
#         text_attention_weights = layer_attentions.text_attention_weights

#         # get sum of attention weights averaged across all heads
#         image_attention_weight_sum = torch.sum(torch.mean(image_attention_weights, dim=1))
#         text_attention_weight_sum = torch.sum(torch.mean(text_attention_weights, dim=1))

#         layer_image_attention_weights.append(image_attention_weight_sum)
#         layer_text_attention_weights.append(text_attention_weight_sum)

#     layer_indices = list(range(len(output_layer_attentions)))

#     fig, ax1 = plt.subplots()

#     color = 'tab:red'
#     ax1.set_xlabel('Layer Index')
#     ax1.set_ylabel('Sum of Attention Weights Given to Image Tokens', color=color)
#     ax1.plot(layer_indices, layer_image_attention_weights, color=color)
#     ax1.tick_params(axis='y', labelcolor=color)

#     ax2 = ax1.twinx()

#     color = 'tab:blue'
#     ax2.set_ylabel('Sum of Attention Weights Given to Text Tokens', color=color)
#     ax2.plot(layer_indices, layer_text_attention_weights, color=color)
#     ax2.tick_params(axis='y', labelcolor=color)

#     plt.title(f"Comparison of Attention Weights Allocated to Image vs Text Tokens\nImage Name: {image_id}")

#     plt.tight_layout()
#     plt.savefig(output_file_path)
    plt.close(fig)


def evaluate(
    dataloader: DataLoader, 
    model: PaliGemmaForConditionalGeneration, 
    processor: AutoProcessor, 
    dataset_type: str,
    split: str,
    vea_config: VEAConfig,
    best_layers_idxes: list[int],
    device: torch.device,
    aim_run: Run = None
):
    preprocessed_image_size = processor.image_processor.size
    preprocessor_image_mean = processor.image_processor.image_mean
    preprocessor_image_std = processor.image_processor.image_std
    patch_size = model.config.vision_config.patch_size
    num_patches = model.config.vision_config.num_image_tokens
    num_patches_w, num_patches_h = [int(num_patches ** (1 / 2))] * 2

    experiment_name = vea_config.experiment_name
    experiment_output_paths = get_experiment_output_paths(args.model_id, dataset_type, split, experiment_name)

    num_correct = 0
    total = 0
    prediction_results = []
    for i, input_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        captions, labels, images, image_names, _, _ = input_data
        encodings = processor(text=captions, images=images, return_tensors="pt", padding="longest", device=device)

        label = labels[0]
        caption = captions[0]
        image = images[0]
        image_name = image_names[0]

        h, w, _ = image.shape

        image_token_indices = torch.where(encodings["input_ids"] == model.config.image_token_index, 1, 0)

        with torch.inference_mode():
            model_outputs = model.generate(
                **encodings, 
                image_token_indices=image_token_indices, 
                max_new_tokens=1, 
                use_cache=True, 
                output_attentions=True, 
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        first_output_token_idx = 0

        # Attention Extraction
        evidence_scores = sum([model_outputs.attentions[first_output_token_idx][layer_idx].image_attention_weights for layer_idx in best_layers_idxes]) / len(best_layers_idxes)
        evidence_scores = torch.mean(evidence_scores, dim=1)
        evidence_scores = evidence_scores.reshape(num_patches_h, num_patches_w)

        # Attention Mask Denoising
        e_denoised = denoise_attention_mask(evidence_scores.numpy(), (num_patches_h, num_patches_w))

        # Highlight Mask Smoothing
        e_smoothed = smooth_highlight_mask(
            e_denoised, 
            grid_hw=(num_patches_h, num_patches_w),
            image_short_side=min(preprocessed_image_size["height"], preprocessed_image_size["width"]),
            patch_size=patch_size,
            smooth_strength=vea_config.smooth_strength,
            return_grid=True,
        )
        e_smoothed = e_smoothed.repeat(patch_size, axis=0).repeat(patch_size, axis=1)

        scale = vea_config.highlight_strength + (1.0 - vea_config.highlight_strength) * e_smoothed  # (H, W)

        pixel_values = encodings['pixel_values']
        c = len(preprocessor_image_mean)
        mean = torch.tensor(preprocessor_image_mean, device=pixel_values.device, dtype=pixel_values.dtype).view(1, c, 1, 1)
        std  = torch.tensor(preprocessor_image_std,  device=pixel_values.device, dtype=pixel_values.dtype).view(1, c, 1, 1)

        # denormalize and apply scale to pixel_values
        denormalized_pixel_values = pixel_values * std + mean
        denormalized_pixel_values_aug = denormalized_pixel_values * scale[None, None, :, :]  # broadcast to (1, 3, H, W)

        # re-normalize
        pixel_values_aug = (denormalized_pixel_values_aug - mean) / std
        encodings['pixel_values'] = pixel_values_aug

        with torch.inference_mode():
            model_outputs = model.generate(
                **encodings, 
                image_token_indices=image_token_indices, 
                max_new_tokens=1, 
                use_cache=True, 
                output_attentions=True, 
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        plot_layer_attentions(
            output_layer_attentions=model_outputs.attentions[first_output_token_idx], 
            image_name=image_name, 
            output_file_path=experiment_output_paths.attention_plots_dir_path / image_name
        )

        final_layer_image_attention_weights = model_outputs.attentions[first_output_token_idx][-1].image_attention_weights
        image_attn_weights_avg_heads = torch.mean(final_layer_image_attention_weights, dim=1)
        image_attn_weights_avg_heads = image_attn_weights_avg_heads.squeeze().reshape(num_patches_h, num_patches_w)

        plt.imsave(
            fname=experiment_output_paths.attention_weights_dir_path / image_name, 
            arr=image_attn_weights_avg_heads, 
            format='jpg'
        )

        # rescale attention weights to original image dimensions
        rescaled_avg_attn_weights = get_rescaled_avg_attn_weights(
            input_image_path=experiment_output_paths.attention_weights_dir_path / image_name,
            image_h=h,
            image_w=w,
            model_patch_size=patch_size
        )
        save_attention_image_overlay(
            img=image, 
            attn_heatmap=rescaled_avg_attn_weights, 
            output_file_path=experiment_output_paths.heatmaps_dir_path / image_name
        )

        decoded_output = processor.decode(model_outputs.sequences[first_output_token_idx], skip_special_tokens=True)[len(captions[0]):].strip()
        confidence = np.round(float(max(torch.softmax(model_outputs.scores[first_output_token_idx].squeeze(), dim=0))), 2)

        if decoded_output == label:
            num_correct += 1
        total += 1

        prediction_result = {
            "image_filename": image_name,
            "prompt": caption,
            "label": label,
            "prediction": decoded_output,
            "confidence": confidence,
        }
        prediction_results.append(prediction_result)

        print(f"Example {i} - Image: {image_name}")
        print(f"\tLabel: {label}, Prediction: {decoded_output}")

        if aim_run:
            track_aim_run(experiment_output_paths=experiment_output_paths, prediction_result=prediction_result, step=i, context={"method": vea_config.method})

    accuracy = num_correct / total

    eval_results = {
        "num_correct": num_correct, 
        "total": total, 
        "accuracy": accuracy, 
        "prediction_results": prediction_results
    }

    if aim_run:
        aim_run.track({k: v for k, v in eval_results.items() if k != "prediction_results"})

    with open(experiment_output_paths.experiment_dir_path / experiment_output_paths.predictions_file_name, "w") as f:
        json.dump(eval_results, f)

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate PaliGemma2 with temperature scaling')
    parser.add_argument('--dataset_type', type=str, choices=["random", "zeroshot"], required=True)
    parser.add_argument('--dataset_split', type=str, choices=["dev", "test"], required=True)
    parser.add_argument('--vsr_data_dir', type=str, required=True)
    parser.add_argument('--coco_annotations_dir', type=str, required=False)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--base_model_id', type=str, default="paligemma2-3b-pt-224")
    parser.add_argument('--smooth_strength', type=float, default=0.5)
    parser.add_argument('--highlight_strength', type=float, default=0.5)
    parser.add_argument('--log_with_aim', action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_type = args.dataset_type
    dataset_split = args.dataset_split

    train_json_path = f"{args.vsr_data_dir}/splits/{dataset_type}/train.jsonl"
    train_dataset = VSRDataset(f"{args.vsr_data_dir}/images", train_json_path, args.coco_annotations_dir)
    val_json_path = f"{args.vsr_data_dir}/splits/{dataset_type}/{dataset_split}.jsonl"
    eval_dataset = VSRDataset(f"{args.vsr_data_dir}/images", val_json_path)

    model_path = f"{MODELS_BASE_DIR}/{args.model_id}"
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, attn_implementation='eager').to(device)
    processor = AutoProcessor.from_pretrained(f"{PRETRAINED_MODELS_BASE_DIR}/{args.base_model_id}")

    # def collate_fn(examples):
    #     captions = ["answer en " + example["caption"] for example in examples]
    #     labels= ["True" if example["label"] else "False" for example in examples]
    #     images = [example["image"] for example in examples]
    #     image_names = [example["image_name"] for example in examples]
    #     image_links = [example["image_link"] for example in examples]
    #     entities = [example["entities"] for example in examples if example is not None]

    #     tokens = processor(text=captions, images=images, return_tensors="pt", padding="longest")
    #     tokens = tokens.to(device)

    #     entities = entities if len(entities) > 0 else None

    #     return tokens, captions, labels, images, image_names, image_links, entities

    eval_dataloader = DataLoader(eval_dataset, collate_fn=paligemma_data_collator, batch_size=1, shuffle=False)
    train_dataloader = DataLoader(train_dataset, collate_fn=paligemma_data_collator, batch_size=1, shuffle=False)

    vea_config = VEAConfig(
        smooth_strength=args.smooth_strength,
        highlight_strength=args.highlight_strength,
    )

    aim_run = None
    if args.log_with_aim:
        aim_run = Run(experiment=f"evaluating_{args.model_id}")

        aim_run["hparams"] = {
            "dataset_type": args.dataset_type,
            "dataset_split": args.dataset_split,
            "smooth_strength": args.smooth_strength,
            "highlight_strength": args.highlight_strength,
        }

    # best_layers_idxes = get_best_layers(
    #     train_dataloader, 
    #     model, 
    #     processor, 
    #     args.coco_annotations_dir, 
    #     device
    # )
    best_layers_idxes = [2, 5, 12]

    accuracy = evaluate(
        eval_dataloader, 
        model, 
        processor, 
        dataset_type, 
        dataset_split, 
        vea_config,
        best_layers_idxes, 
        device,
        aim_run
    )

    print("Accuracy:", accuracy)
