import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from aim import Run
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, set_seed

from visual_attention_intervention.base import (
    AdaptVisConfig,
    ExperimentConfig,
    MODELS_BASE_DIR, 
    Method, 
    PRETRAINED_MODELS_BASE_DIR,
    get_experiment_output_paths, 
)
from visual_attention_intervention.modeling_paligemma2 import PaliGemmaForConditionalGeneration
from visual_attention_intervention.utils import (
    paligemma_data_collator,
    plot_layer_attentions,
    save_attention_image_overlay,
    track_aim_run, 
)
from visual_attention_intervention.vsr_dataset import VSRDataset

torch.manual_seed(42)
set_seed(42)


def get_rescaled_avg_attn_weights(input_image_path, image_h, image_w, model_patch_size):
    avg_attn_weights = plt.imread(input_image_path)
    avg_attn_weights = np.transpose(avg_attn_weights, (2, 0, 1))

    rescaled_avg_attn_weights = F.interpolate(
        torch.from_numpy(avg_attn_weights.copy()).unsqueeze(0), scale_factor=model_patch_size, mode='bicubic'
    )[0]
    rescaled_avg_attn_weights = F.interpolate(
        rescaled_avg_attn_weights.unsqueeze(0), (image_h, image_w), mode='bilinear'
    )[0].cpu().numpy()   # (c, h, w)
    rescaled_avg_attn_weights = np.transpose(rescaled_avg_attn_weights, (1, 2, 0))    # (h, w, c)

    return rescaled_avg_attn_weights


# def save_attention_image_overlay(img, attn_heatmap, output_file_path, alpha=0.5):
#     fig, _ = plt.subplots()
#     plt.imshow(img)
#     plt.imshow(attn_heatmap, alpha=alpha)
#     plt.axis('off')
#     plt.savefig(output_file_path, bbox_inches="tight")
#     plt.close(fig)


# def plot_layer_attentions(output_layer_attentions, image_name, output_file_path, weight=None):
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

#     title = f"Comparison of Attention Weights Allocated to Image vs Text Tokens\nImage Name: {image_name}"
#     if weight is not None:
#         title += f", Image Weight: {weight}"
#     plt.title(title)

#     plt.tight_layout()
#     plt.savefig(output_file_path)
#     plt.close(fig)


def evaluate(
    dataloader: DataLoader, 
    model: PaliGemmaForConditionalGeneration, 
    processor: AutoProcessor, 
    dataset_type: str,
    split: str,
    standard_config: ExperimentConfig,
    adapt_vis_config: AdaptVisConfig,
    device: torch.device,
    aim_run: Run = None
):
    patch_size = model.config.vision_config.patch_size
    num_patches = model.config.vision_config.num_image_tokens
    num_patches_w, num_patches_h = [int(num_patches ** (1 / 2))] * 2

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
        captions, labels, images, image_names, _, _ = input_data
        encodings = processor(text=captions, images=images, return_tensors="pt", padding="longest", device=device)

        label = labels[0]
        caption = captions[0]
        image = images[0]
        image_name = image_names[0]

        h, w, _ = image.shape

        image_token_indices = torch.where(encodings["input_ids"] == model.config.image_token_index, 1, 0)

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
                image_name=image_name, 
                output_file_path=experiment_output_paths.attention_plots_dir_path / file_name,
                weight=weight, 
            )

            final_layer_image_attention_weights = model_outputs.attentions[first_output_token_idx][-1].image_attention_weights
            image_attn_weights_avg_heads = torch.mean(final_layer_image_attention_weights, dim=1)
            image_attn_weights_avg_heads = image_attn_weights_avg_heads.squeeze().reshape(num_patches_h, num_patches_w)

            plt.imsave(
                fname=experiment_output_paths.attention_weights_dir_path / file_name, 
                arr=image_attn_weights_avg_heads, 
                format='jpg'
            )

            # rescale attention weights to original image dimensions
            rescaled_avg_attn_weights = get_rescaled_avg_attn_weights(
                input_image_path=experiment_output_paths.attention_weights_dir_path / file_name,
                image_h=h,
                image_w=w,
                model_patch_size=patch_size
            )
            save_attention_image_overlay(
                img=image, 
                attn_heatmap=rescaled_avg_attn_weights, 
                output_file_path=experiment_output_paths.heatmaps_dir_path / file_name
            )

            decoded_output = processor.decode(
                model_outputs.sequences[first_output_token_idx], skip_special_tokens=True
            )[len(captions[0]):].strip()
            confidence = np.round(float(max(torch.softmax(model_outputs.scores[first_output_token_idx].squeeze(), dim=0))), 2)

            prediction_result = {
                "image_filename": file_name,
                "prompt": caption,
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
    parser = argparse.ArgumentParser(description='Evaluate PaliGemma2 on VSR with AdaptVis')
    parser.add_argument('--dataset_type', type=str, choices=["random", "zeroshot"], required=True)
    parser.add_argument('--dataset_split', type=str, choices=["dev", "test"], required=True)
    parser.add_argument('--vsr_data_dir', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--base_model_id', type=str, default="paligemma2-3b-pt-224")
    parser.add_argument('--confidence_threshold', type=float, default=0.9)
    parser.add_argument('--sharpen_weight', type=float, default=2.0)
    parser.add_argument('--smoothen_weight', type=float, default=0.08)
    parser.add_argument('--log_with_aim', action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_type = args.dataset_type
    dataset_split = args.dataset_split

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

    #     tokens = processor(text=captions, images=images, return_tensors="pt", padding="longest")
    #     tokens = tokens.to(device)

    #     return tokens, captions, labels, images, image_names

    dataloader = DataLoader(eval_dataset, collate_fn=paligemma_data_collator, batch_size=1, shuffle=False)

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
