import argparse
import cv2
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from aim import Run
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen3VLProcessor, set_seed

from visual_attention_intervention.base import (
    AdaptVisConfig,
    ExperimentConfig,
    MODELS_BASE_DIR, 
    Method, 
    PRETRAINED_MODELS_BASE_DIR,
    get_experiment_output_paths, 
)
from visual_attention_intervention.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from visual_attention_intervention.utils import (
    plot_layer_attentions,
    qwen_data_collator,
    save_attention_image_overlay,
    track_aim_run, 
)
from visual_attention_intervention.vsr_dataset import VSRDataset

torch.manual_seed(42)
set_seed(42)


def evaluate(
    dataloader: DataLoader, 
    model: Qwen3VLForConditionalGeneration, 
    processor: Qwen3VLProcessor, 
    dataset_type: str,
    split: str,
    standard_config: ExperimentConfig,
    adapt_vis_config: AdaptVisConfig,
    device: torch.device,
    aim_run: Run = None
):
    patch_size = model.config.vision_config.patch_size
    spatial_merge_size = model.config.vision_config.spatial_merge_size

    method_to_exp = {
        Method.STANDARD: get_experiment_output_paths(args.model_id, dataset_type, split, standard_config.experiment_name),
        Method.ADAPT_VIS: get_experiment_output_paths(args.model_id, dataset_type, split, adapt_vis_config.experiment_name),
    }
    method_to_results = {
        method: {"num_correct": 0, "total": 0, "accuracy": 0, "prediction_results": []} 
        for method in method_to_exp
    }

    sharpen_count = 0
    smoothen_count = 0
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

        image_token_indices = (encodings["input_ids"][0] == model.config.image_token_id).nonzero(as_tuple=True)[0]

        weight = 1.0
        description = ""
        for method in method_to_exp:
            print(f"Running inference using {method.value} method.")
            experiment_output_paths = method_to_exp[method]

            file_name = image_name.replace(".jpg", f"{description}.jpg")

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
                )

            first_output_token_idx = 0  # True/False output
            plot_layer_attentions(
                output_layer_attentions=model_outputs.attentions[first_output_token_idx], 
                image_name=file_name, 
                output_file_path=experiment_output_paths.attention_plots_dir_path / file_name,
                weight=weight, 
            )

            final_layer_image_attention_weights = model_outputs.attentions[first_output_token_idx][-1].image_attention_weights
            image_attn_weights_avg_heads = torch.mean(final_layer_image_attention_weights, dim=1)
            image_attn_weights_avg_heads = image_attn_weights_avg_heads.squeeze().reshape(
                num_patches_h // spatial_merge_size, num_patches_w // spatial_merge_size
            )

            plt.imsave(
                fname=experiment_output_paths.attention_weights_dir_path / file_name, 
                arr=image_attn_weights_avg_heads, 
                format='jpg',
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
                output_file_path=experiment_output_paths.heatmaps_dir_path / file_name,
            )

            prompt_len = encodings["input_ids"].shape[1]
            new_tokens = model_outputs.sequences[first_output_token_idx][prompt_len:]
            decoded_outputs = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            decoded_output = decoded_outputs[0]

            confidence = np.round(float(max(torch.softmax(model_outputs.scores[first_output_token_idx].squeeze(), dim=0))), 2)

            prediction_result = {
                "image_filename": file_name,
                "prompt": prompt,
                "label": label,
                "prediction": decoded_output,
                "confidence": confidence,
            }
            method_to_results[method]["prediction_results"].append(prediction_result)

            if decoded_output == label:
                method_to_results[method]["num_correct"] += 1
            method_to_results[method]["total"] += 1

            method_to_results[method]["accuracy"] = (
                method_to_results[method]["num_correct"] / method_to_results[method]["total"]
            )

            # get model outputs with temperature scaling based on confidence of original model output
            if confidence < adapt_vis_config.threshold:
                weight = adapt_vis_config.smoothen_weight
                smoothen_count += 1
                description = f"_{adapt_vis_config.smoothen_description}"
            else:
                weight = adapt_vis_config.sharpen_weight
                sharpen_count += 1
                description = f"_{adapt_vis_config.sharpen_description}"

            print(f"Example {i} - Image: {image_name}")
            print(f"\tLabel: {label}, Prediction: {decoded_output}")

            if aim_run:
                track_aim_run(experiment_output_paths=experiment_output_paths, prediction_result=prediction_result, step=i, context={"method": method.value})

    method_to_results[Method.ADAPT_VIS]["num_sharpen_examples"] = sharpen_count
    method_to_results[Method.ADAPT_VIS]["num_smoothen_examples"] = smoothen_count

    for method in method_to_exp:
        experiment_output_paths = method_to_exp[method]

        results = method_to_results[method]

        with open(experiment_output_paths.experiment_dir_path / experiment_output_paths.predictions_file_name, "w") as f:
            json.dump(results, f)

        if aim_run:
            aim_run.track({k: v for k, v in results.items() if k != "prediction_results"}, context={'method': method.value})

    return method_to_results[Method.ADAPT_VIS]["accuracy"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Qwen3-VL on VSR with AdaptVis')
    parser.add_argument('--dataset_type', type=str, choices=["random", "zeroshot"], required=True)
    parser.add_argument('--dataset_split', type=str, choices=["dev", "test"], required=True)
    parser.add_argument('--vsr_data_dir', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--base_model_id', type=str, default="Qwen3-VL-2B-Instruct")
    parser.add_argument('--confidence_threshold', type=float, default=0.8)
    parser.add_argument('--sharpen_weight', type=float, default=1.2)
    parser.add_argument('--smoothen_weight', type=float, default=0.2)
    parser.add_argument('--log_with_aim', action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_type = args.dataset_type
    dataset_split = args.dataset_split

    val_json_path = f"{args.vsr_data_dir}/splits/{dataset_type}/{dataset_split}.jsonl"
    eval_dataset = VSRDataset(f"{args.vsr_data_dir}/images", val_json_path)

    model_path = f"{MODELS_BASE_DIR}/{args.model_id}"
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, attn_implementation='eager').to(device)
    processor = Qwen3VLProcessor.from_pretrained(f"{PRETRAINED_MODELS_BASE_DIR}/{args.base_model_id}")

    dataloader = DataLoader(eval_dataset, collate_fn=qwen_data_collator, batch_size=1, shuffle=False)

    standard_config = ExperimentConfig()
    adapt_vis_config = AdaptVisConfig(
        threshold=args.confidence_threshold,
        sharpen_weight=args.sharpen_weight,
        smoothen_weight=args.smoothen_weight,
    )

    aim_run = None
    if args.log_with_aim:
        aim_run = Run(experiment=f"evaluating_{args.model_id}")

        aim_run["hparams"] = {
            "dataset_type": args.dataset_type,
            "dataset_split": args.dataset_split,
            "threshold": args.confidence_threshold,
            "sharpen_weight": args.sharpen_weight,
            "smoothen_weight": args.smoothen_weight,
        }

    accuracy = evaluate(
        dataloader, 
        model, 
        processor, 
        dataset_type, 
        dataset_split, 
        standard_config,
        adapt_vis_config,
        device,
        aim_run
    )

    print("Accuracy:", accuracy)
