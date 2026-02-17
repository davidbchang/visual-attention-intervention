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
    CLVSConfig,
    MODELS_BASE_DIR, 
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
    clvs_config: CLVSConfig,
    device: torch.device,
    aim_run: Run = None,
):
    patch_size = model.config.vision_config.patch_size
    spatial_merge_size = model.config.vision_config.spatial_merge_size

    experiment_name = clvs_config.experiment_name
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

        image_token_indices = (encodings["input_ids"][0] == model.config.image_token_id).nonzero(as_tuple=True)[0]

        with torch.inference_mode():
            model_outputs = model.generate(
                **encodings, 
                image_token_indices=image_token_indices, 
                use_clvs=True,
                max_new_tokens=1, 
                use_cache=True, 
                output_attentions=True, 
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                smoothing=clvs_config.smoothing,
                window_memory_size=clvs_config.window_memory_size,
                uncertainty_threshold=clvs_config.uncertainty_threshold,
            )

        first_output_token_idx = 0
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
            "caption": prompt,
            "label": label,
            "prediction": decoded_output,
            "confidence": confidence,
        }
        prediction_results.append(prediction_result)

        print(f"Example {i} - Image: {image_name}")
        print(f"\tLabel: {label}, Prediction: {decoded_output}")

        if aim_run:
            track_aim_run(experiment_output_paths=experiment_output_paths, prediction_result=prediction_result, step=i, context={"method": clvs_config.method})

    accuracy = num_correct / total

    eval_results = {
        "num_correct": num_correct, 
        "total": total, 
        "accuracy": accuracy, 
        "prediction_results": prediction_results
    }

    with open(experiment_output_paths.experiment_dir_path / experiment_output_paths.predictions_file_name, "w") as f:
        json.dump(eval_results, f)

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Qwen3-VL')
    parser.add_argument('--dataset_type', type=str, choices=["random", "zeroshot"], required=True)
    parser.add_argument('--dataset_split', type=str, choices=["dev", "test"], required=True)
    parser.add_argument('--vsr_data_dir', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--base_model_id', type=str, default="Qwen3-VL-2B-Instruct")
    parser.add_argument('--smoothing', type=float, default=0.8)
    parser.add_argument('--window_memory_size', type=float, default=0.8)
    parser.add_argument('--uncertainty_threshold', type=float, default=0.5)
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

    clvs_config = CLVSConfig(
        smoothing=args.smoothing,
        window_memory_size=args.window_memory_size,
        uncertainty_threshold=args.uncertainty_threshold,
    )

    aim_run = None
    if args.log_with_aim:
        aim_run = Run(experiment=f"evaluating_{args.model_id}")

        aim_run["hparams"] = {
            "dataset_type": args.dataset_type,
            "dataset_split": args.dataset_split,
            "smoothing": args.smoothing,
            "window_memory_size": args.window_memory_size,
            "uncertainty_threshold": args.uncertainty_threshold,
        }

    accuracy = evaluate(
        dataloader, 
        model, 
        processor, 
        dataset_type, 
        dataset_split, 
        clvs_config,
        device,
        aim_run
    )

    print("Accuracy:", accuracy)
