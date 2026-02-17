import argparse

import torch
from aim.hugging_face import AimCallback
from peft import get_peft_model, LoraConfig
from transformers import (
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.utils import is_flash_attn_2_available

from visual_attention_intervention.base import MODELS_BASE_DIR, PRETRAINED_MODELS_BASE_DIR
from visual_attention_intervention.modeling_paligemma2 import PaliGemmaForConditionalGeneration
from visual_attention_intervention.vsr_dataset import VSRDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune PaliGemma2 on VSR')
    parser.add_argument('--dataset_type', type=str, choices=["random", "zeroshot"], required=True)
    parser.add_argument('--vsr_data_dir', type=str, required=True)
    parser.add_argument('--base_model_id', type=str, default="paligemma2-3b-pt-224")
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=256)
    parser.add_argument('--freeze_vision', action='store_true')
    parser.add_argument('--freeze_projector', action='store_true')
    parser.add_argument('--log_with_aim', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_feature_path = f"{args.vsr_data_dir}/images"
    train_json_path = f"{args.vsr_data_dir}/splits/{args.dataset_type}/train.jsonl"
    val_json_path = f"{args.vsr_data_dir}/splits/{args.dataset_type}/dev.jsonl"

    dataset_train = VSRDataset(img_feature_path, train_json_path)
    dataset_val = VSRDataset(img_feature_path, val_json_path)

    num_train_data = len(dataset_train)

    model_id = f"{PRETRAINED_MODELS_BASE_DIR}/{args.base_model_id}"

    processor = PaliGemmaProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "eager", dtype=torch.bfloat16
    ).to(device)
    model.config.use_cache = False

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    for param in model.vision_tower.parameters():
        param.requires_grad = not args.freeze_vision

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = not args.freeze_projector

    model_name = f"{args.base_model_id}_causal-lm{'_freeze-vision' if args.freeze_vision else ''}{'_freeze-projector' if args.freeze_projector else ''}_vsr"
    model_name += f"_b{args.train_batch_size}x{args.gradient_accumulation_steps}"
    model_name += f"_{args.learning_rate}_e{args.num_train_epochs}"
    model_name += f"_{args.dataset_type}-train-{num_train_data}"
    if args.use_lora:
        model_name += f"_lora-r{args.lora_r}-a{args.lora_alpha}"
    output_dir = f"{MODELS_BASE_DIR}/{model_name}_testing_new"

    training_args=TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        remove_unused_columns=False,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=5,
        learning_rate=args.learning_rate,
        weight_decay=1e-6,
        adam_beta2=0.999,
        label_names=["labels"],
        logging_strategy="steps",
        logging_steps=1,
        log_level="debug",
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=20,
        save_steps=20,
        save_strategy="best",
        save_total_limit=1,
        save_only_model=True,
        push_to_hub=False,
        bf16=True,
        dataloader_pin_memory=False,
        use_cpu=device.type == "cpu",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    def collate_fn(examples):
        captions = ["answer en " + example["caption"] for example in examples]
        labels= ["True" if example["label"] else "False" for example in examples]
        images = [example["image"] for example in examples]

        tokens = processor(
            text=captions, 
            images=images, 
            suffix=labels, 
            return_tensors="pt", 
            padding="longest"
        )
        tokens = tokens.to(device)
        return tokens

    trainer = Trainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_fn,
        callbacks=[AimCallback(experiment=model_name)] if args.log_with_aim else None,
        args=training_args
    )

    trainer.train()
