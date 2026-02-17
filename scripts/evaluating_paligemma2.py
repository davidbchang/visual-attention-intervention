import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, set_seed

from visual_attention_intervention.base import (
    ExperimentConfig, 
    MODELS_BASE_DIR, 
    PRETRAINED_MODELS_BASE_DIR,
    get_experiment_output_paths, 
)
from visual_attention_intervention.modeling_paligemma2 import PaliGemmaForConditionalGeneration
from visual_attention_intervention.utils import paligemma_data_collator
from visual_attention_intervention.vsr_dataset import VSRDataset

torch.manual_seed(42)
set_seed(42)


def evaluate(
    dataloader: DataLoader, 
    model: PaliGemmaForConditionalGeneration, 
    processor: AutoProcessor, 
    dataset_type: str,
    split: str,
    device: torch.device,
):
    experiment_name = ExperimentConfig().experiment_name
    experiment_output_paths = get_experiment_output_paths(args.model_id, dataset_type, split, experiment_name)

    num_correct = 0
    total = 0
    prediction_results = []
    for i, input_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        captions, labels, images, image_names, _, _ = input_data
        encodings = processor(text=captions, images=images, return_tensors="pt", padding="longest", device=device)

        with torch.inference_model():
            model_outputs = model.generate(
                **encodings, 
                max_new_tokens=1, 
                do_sample=False,
                use_cache=True,
            )

        for output, caption, label, image_name in zip(model_outputs, captions, labels, image_names):
            decoded_output = processor.decode(output, skip_special_tokens=True)[len(caption):].strip()

            print(f"Example {i} - Image: {image_name}")
            print(f"\tLabel: {label}, Prediction: {decoded_output}")

            if decoded_output == label:
                num_correct += 1
            total += 1

            prediction_results.append({
                "image_filename": image_name,
                "caption": caption,
                "label": label,
                "prediction": decoded_output
            })

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
    parser = argparse.ArgumentParser(description='Evaluate PaliGemma2 on VSR')
    parser.add_argument('--dataset_type', type=str, choices=["random", "zeroshot"], required=True)
    parser.add_argument('--dataset_split', type=str, choices=["dev", "test"], required=True)
    parser.add_argument('--vsr_data_dir', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--base_model_id', type=str, default="paligemma2-3b-pt-224")
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

    #     return tokens, captions, labels, image_names

    dataloader = DataLoader(eval_dataset, collate_fn=paligemma_data_collator, batch_size=1, shuffle=False)

    accuracy = evaluate(dataloader, model, processor, dataset_type, dataset_split, device)

    print("Accuracy:", accuracy)
