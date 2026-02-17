import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from aim import Run
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from visual_attention_intervention.base import (
    CLVSConfig, 
    get_experiment_output_paths, 
    MODELS_BASE_DIR, 
    PRETRAINED_MODELS_BASE_DIR
)
from visual_attention_intervention.modeling_paligemma2 import PaliGemmaForConditionalGeneration
from visual_attention_intervention.utils import (
    paligemma_data_collator,
    plot_layer_attentions,
    save_attention_image_overlay,
    track_aim_run, 
)
from visual_attention_intervention.vsr_dataset import VSRDataset


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
#     plt.close(fig)


def evaluate(
    dataloader: DataLoader, 
    model: PaliGemmaForConditionalGeneration, 
    processor: AutoProcessor, 
    dataset_type: str,
    split: str,
    clvs_config: CLVSConfig,
    device: torch.device,
    aim_run: Run = None
):
    preprocessed_image_size = processor.image_processor.size
    preprocessor_image_mean = processor.image_processor.image_mean
    preprocessor_image_std = processor.image_processor.image_std
    patch_size = model.config.vision_config.patch_size
    num_patches = model.config.vision_config.num_image_tokens
    num_patches_w, num_patches_h = [int(num_patches ** (1 / 2))] * 2

    experiment_name = clvs_config.experiment_name
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
                use_clvs=True,
                max_new_tokens=1, 
                use_cache=False, 
                output_attentions=True, 
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                smoothing=0.7,
                window_memory_size=0.8,
                uncertainty_threshold=0.5,
            )

        first_output_token_idx = 0
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
            "caption": caption,
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

    if aim_run:
        aim_run.track({k: v for k, v in eval_results.items() if k != "prediction_results"})

    with open(experiment_output_paths.experiment_dir_path / experiment_output_paths.predictions_file_name, "w") as f:
        json.dump(eval_results, f)

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate PaliGemma2')
    parser.add_argument('--dataset_type', type=str, choices=["random", "zeroshot"], required=True)
    parser.add_argument('--dataset_split', type=str, choices=["dev", "test"], required=True)
    parser.add_argument('--vsr_data_dir', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--base_model_id', type=str, default="paligemma2-3b-pt-224")
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
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, attn_implementation='eager').to(device)
    processor = AutoProcessor.from_pretrained(f"{PRETRAINED_MODELS_BASE_DIR}/{args.base_model_id}")

    # def collate_vsr_fn(examples):
    #     captions = ["answer en " + example["caption"] for example in examples]
    #     labels= ["True" if example["label"] else "False" for example in examples]
    #     images = [example["image"] for example in examples]
    #     image_names = [example["image_name"] for example in examples]
    #     image_links = [example["image_link"] for example in examples]
    #     entities = [example["entities"] for example in examples]

    #     tokens = processor(text=captions, images=images, return_tensors="pt", padding="longest")
    #     tokens = tokens.to(device)

    #     return tokens, captions, labels, images, image_names, image_links, entities

    dataloader = DataLoader(eval_dataset, collate_fn=paligemma_data_collator, batch_size=1, shuffle=False)

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
