import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen3VLProcessor, set_seed

from visual_attention_intervention.base import (
    ExperimentConfig, 
    MODELS_BASE_DIR, 
    PRETRAINED_MODELS_BASE_DIR,
    get_experiment_output_paths, 
)
from visual_attention_intervention.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from visual_attention_intervention.utils import qwen_data_collator
from visual_attention_intervention.vsr_dataset import VSRDataset

torch.manual_seed(42)
set_seed(42)


def evaluate(
    dataloader: DataLoader, 
    model: Qwen3VLForConditionalGeneration, 
    processor: Qwen3VLProcessor, 
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
        prompts, labels, images, image_names, _, _ = input_data

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

        with torch.inference_mode():
            model_outputs = model.generate(
                **encodings, 
                max_new_tokens=1, 
                do_sample=False,
                use_cache=True,
            )

        prompt_len = encodings["input_ids"].shape[1]
        new_tokens = model_outputs[:, prompt_len:]
        decoded_outputs = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        decoded_output = decoded_outputs[0]

        for decoded_output, caption, label, image_name in zip(decoded_outputs, prompts, labels, image_names):
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
    parser = argparse.ArgumentParser(description='Evaluate Qwen3-VL on VSR')
    parser.add_argument('--dataset_type', type=str, choices=["random", "zeroshot"], required=True)
    parser.add_argument('--dataset_split', type=str, choices=["dev", "test"], required=True)
    parser.add_argument('--vsr_data_dir', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--base_model_id', type=str, default="Qwen3-VL-2B-Instruct")
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

    accuracy = evaluate(dataloader, model, processor, dataset_type, dataset_split, device)

    print("Accuracy:", accuracy)
