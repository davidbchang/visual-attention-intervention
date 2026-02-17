import argparse
import cv2
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
from transformers import Qwen3VLProcessor, set_seed

from visual_attention_intervention.base import (
    VEAConfig,
    get_experiment_output_paths, 
    MODELS_BASE_DIR, 
    PRETRAINED_MODELS_BASE_DIR
)
from visual_attention_intervention.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from visual_attention_intervention.utils import (
    compute_auroc,
    compute_patch_labels,
    denoise_attention_mask, 
    plot_layer_attentions,
    qwen_data_collator,
    save_attention_image_overlay,
    smooth_highlight_mask,
    track_aim_run,
)
from visual_attention_intervention.vsr_dataset import VSRDataset

torch.manual_seed(42)
set_seed(42)


def get_best_layers(
    dataloader: DataLoader, 
    model: Qwen3VLForConditionalGeneration, 
    processor: Qwen3VLProcessor,
    coco_annotations_dir: str,
    device: torch.device,
    data_subset_size: int = 100,
    top_k: float = 0.1,
):
    num_hidden_layers = model.config.text_config.num_hidden_layers
    patch_size = model.config.vision_config.patch_size
    spatial_merge_size = model.config.vision_config.spatial_merge_size

    auroc_sum_per_layer = [0] * num_hidden_layers

    coco_data_val = COCO(f"{coco_annotations_dir}/instances_val2017.json")
    coco_data_train = COCO(f"{coco_annotations_dir}/instances_train2017.json")

    num_examples_processed = 0
    for input_data in tqdm(dataloader, total=data_subset_size):
        if num_examples_processed == data_subset_size:
            break

        prompts, labels, images, image_names, image_links, entities = input_data

        prompt_text = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in prompts]
        processor.tokenizer.padding_side = "left"
        encodings = processor(
            text=prompt_text,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            return_token_type_ids=False,
            device=device,
        )
        _, num_patches_h, num_patches_w = [grid_dim.item() for grid_dim in encodings["image_grid_thw"][0]]

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
            image_size=(num_patches_w * patch_size, num_patches_h * patch_size), 
            patch_size=patch_size * spatial_merge_size, 
            bbox_source_size=(w, h)
        )
        if len(np.unique(np.asarray(patch_labels))) < 2:
            print(f"Only 1 entity found for image {image_name}, skipping example")
            continue

        image_token_indices = (encodings["input_ids"][0] == model.config.image_token_id).nonzero(as_tuple=True)[0]

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


def evaluate(
    dataloader: DataLoader, 
    model: Qwen3VLForConditionalGeneration, 
    processor: Qwen3VLProcessor, 
    dataset_type: str,
    split: str,
    vea_config: VEAConfig,
    best_layers_idxes: list[int],
    device: torch.device,
    aim_run: Run = None
):
    patch_size = model.config.vision_config.patch_size
    spatial_merge_size = model.config.vision_config.spatial_merge_size
    temporal_patch_size = model.config.vision_config.temporal_patch_size

    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    experiment_name = vea_config.experiment_name
    experiment_output_paths = get_experiment_output_paths(args.model_id, dataset_type, split, experiment_name)

    num_correct = 0
    total = 0
    prediction_results = []
    for i, input_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        prompts, labels, images, image_names, _, _ = input_data

        label = labels[0]
        prompt = prompts[0]
        image = images[0]
        image_name = image_names[0]

        prompt_text = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in prompts]
        processor.tokenizer.padding_side = "left"
        encodings = processor(
            text=prompt_text,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            return_token_type_ids=False,
            device=device,
        )
        _, num_patches_h, num_patches_w = [grid_dim.item() for grid_dim in encodings["image_grid_thw"][0]]
        image_h = num_patches_h * patch_size
        image_w = num_patches_w * patch_size

        image_token_indices = (encodings["input_ids"][0] == model.config.image_token_id).nonzero(as_tuple=True)[0]

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
        evidence_scores = evidence_scores.reshape(num_patches_h // spatial_merge_size, num_patches_w // spatial_merge_size)

        # Attention Mask Denoising
        e_denoised = denoise_attention_mask(evidence_scores.numpy(), (num_patches_h // spatial_merge_size, num_patches_w // spatial_merge_size))

        # Highlight Mask Smoothing
        e_smoothed = smooth_highlight_mask(
            e_denoised, 
            grid_hw=(num_patches_h // spatial_merge_size, num_patches_w // spatial_merge_size),
            image_short_side=min(image_h, image_w),
            patch_size=patch_size,
            smooth_strength=vea_config.smooth_strength,
            return_grid=True
        )
        patches_smoothed = e_smoothed.repeat(spatial_merge_size, axis=0).repeat(spatial_merge_size, axis=1)
        patches_smoothed = patches_smoothed.reshape(
            num_patches_h // spatial_merge_size, 
            spatial_merge_size, 
            num_patches_w // spatial_merge_size, 
            spatial_merge_size
        ).transpose(0, 2, 1, 3).reshape(-1)

        scale = vea_config.highlight_strength + (1.0 - vea_config.highlight_strength) * patches_smoothed  # (H,W)
        scale = scale.reshape(-1, 1)

        pixel_values = encodings['pixel_values']
        patch_dim = int(pixel_values.shape[1])

        c = len(image_mean)
        denom = c * patch_size * patch_size
        if patch_dim % denom != 0:
            raise ValueError(f"patch_dim={patch_dim} not divisible by (3 * patch_size * patch_size)={denom}")

        pixel_values = pixel_values.view(num_patches_h * num_patches_w, temporal_patch_size, c, patch_size, patch_size)

        mean = torch.tensor(image_mean, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 1, c, 1, 1)
        std  = torch.tensor(image_std,  device=pixel_values.device, dtype=pixel_values.dtype).view(1, 1, c, 1, 1)

        # denormalize and apply scale to pixel_values
        denormalized_pixel_values = pixel_values * std + mean
        denormalized_pixel_values_aug = denormalized_pixel_values * scale.reshape(-1, 1, 1, 1, 1)

        # re-normalize
        pixel_values_aug = (denormalized_pixel_values_aug - mean) / std
        pixel_values_aug = pixel_values_aug.reshape(num_patches_h * num_patches_w, patch_dim)
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
        image_attn_weights_avg_heads = image_attn_weights_avg_heads.squeeze().reshape(
            num_patches_h // spatial_merge_size, num_patches_w // spatial_merge_size
        )

        plt.imsave(
            fname=experiment_output_paths.attention_weights_dir_path / image_name, 
            arr=image_attn_weights_avg_heads, 
            format='jpg'
        )

        # rescale attention weights to the resized image from the processor
        rescaled_avg_attn_weights = F.interpolate(
            image_attn_weights_avg_heads.unsqueeze(0).unsqueeze(0), 
            scale_factor=patch_size * spatial_merge_size, 
            mode='bicubic'
        )[0]
        rescaled_avg_attn_weights = rescaled_avg_attn_weights.permute(1, 2, 0)

        # resize original image to the resized image from the processor
        image = cv2.resize(image, (num_patches_w * patch_size, num_patches_h * patch_size), interpolation=cv2.INTER_CUBIC)

        save_attention_image_overlay(
            img=image, 
            attn_heatmap=rescaled_avg_attn_weights, 
            output_file_path=experiment_output_paths.heatmaps_dir_path / image_name,
        )

        prompt_len = encodings["input_ids"].shape[1]
        new_tokens = model_outputs.sequences[first_output_token_idx][prompt_len:]
        decoded_outputs = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        decoded_output = decoded_outputs[0]

        confidence = np.round(float(max(torch.softmax(model_outputs.scores[first_output_token_idx].squeeze(), dim=0))), 2)

        if decoded_output == label:
            num_correct += 1
        total += 1

        prediction_result = {
            "image_filename": image_name,
            "prompt": prompt,
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
    parser = argparse.ArgumentParser(description='Evaluate Qwen3 VL with temperature scaling')
    parser.add_argument('--dataset_type', type=str, choices=["random", "zeroshot"], required=True)
    parser.add_argument('--dataset_split', type=str, choices=["dev", "test"], required=True)
    parser.add_argument('--vsr_data_dir', type=str, required=True)
    parser.add_argument('--coco_annotations_dir', type=str, required=False)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--base_model_id', type=str, default="Qwen3-VL-2B-Instruct")
    parser.add_argument('--smooth_strength', type=float, default=0.5)
    parser.add_argument('--highlight_strength', type=float, default=0.5)
    parser.add_argument('--log_with_aim', action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_type = args.dataset_type
    dataset_split = args.dataset_split

    train_json_path = f"{args.vsr_data_dir}/splits/{dataset_type}/train.jsonl"
    train_dataset = VSRDataset(f"{args.vsr_data_dir}/images", train_json_path)
    val_json_path = f"{args.vsr_data_dir}/splits/{dataset_type}/{dataset_split}.jsonl"
    eval_dataset = VSRDataset(f"{args.vsr_data_dir}/images", val_json_path, args.coco_annotations_dir)

    model_path = f"{MODELS_BASE_DIR}/{args.model_id}"
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, attn_implementation='eager').to(device)
    processor = Qwen3VLProcessor.from_pretrained(f"{PRETRAINED_MODELS_BASE_DIR}/{args.base_model_id}")

    eval_dataloader = DataLoader(eval_dataset, collate_fn=qwen_data_collator, batch_size=1, shuffle=False)
    train_dataloader = DataLoader(train_dataset, collate_fn=qwen_data_collator, batch_size=1, shuffle=False)

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
    #     device,
    # )
    best_layers_idxes = [17, 20, 24] if dataset_type == "random" else [17, 20, 10]
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
