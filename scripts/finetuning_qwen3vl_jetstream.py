import argparse

import torch
from aim.hugging_face import AimCallback
from peft import get_peft_model, LoraConfig
from transformers import PaliGemmaProcessor, Qwen3VLProcessor, Qwen3VLForConditionalGeneration
from transformers import Trainer
from transformers import TrainingArguments

from vsr.base import PRETRAINED_MODELS_BASE_DIR
# from paligemma_vsr.modeling_paligemma2 import PaliGemmaForConditionalGeneration
from vsr.vsr_dataset import VSRDataset
# from rich import print


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune LoRA adapterfor PaliGemma2')
    parser.add_argument('--dataset_type', type=str, choices=["random", "zeroshot"], required=True)
    parser.add_argument('--vsr_data_dir', type=str, required=True)
    # parser.add_argument('--base_model_id', type=str, default="paligemma2-3b-pt-224")
    parser.add_argument('--base_model_id', type=str, default="Qwen3-VL-2B-Instruct")
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=8)
    # parser.add_argument('--train_batch_size', type=int, default=2)
    # parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=32)
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=256)
    parser.add_argument('--freeze_vision', action='store_true')
    parser.add_argument('--freeze_projector', action='store_true')
    parser.add_argument('--log_with_aim', action='store_true')
    args = parser.parse_args()

    device = "cuda"

    img_feature_path = f"{args.vsr_data_dir}/images"
    train_json_path = f"{args.vsr_data_dir}/splits/{args.dataset_type}/train.jsonl"
    val_json_path = f"{args.vsr_data_dir}/splits/{args.dataset_type}/dev.jsonl"

    dataset_train = VSRDataset(img_feature_path, train_json_path)
    dataset_val = VSRDataset(img_feature_path, val_json_path)

    num_train_data = len(dataset_train)

    model_id = f"{PRETRAINED_MODELS_BASE_DIR}/{args.base_model_id}"

    # processor = PaliGemmaProcessor.from_pretrained(model_id)
    processor = Qwen3VLProcessor.from_pretrained(model_id)
    # model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, attn_implementation='eager').to(device)
    # model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, attn_implementation='eager').to(device)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, attn_implementation="flash_attention_2", dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    # print(model)
    # exit()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    # print(model)
    # exit()

    # model = get_peft_model(model, lora_config)

    if args.freeze_vision:
        for param in model.visual.parameters():
            param.requires_grad = False

    # for param in model.multi_modal_projector.parameters():
    #     param.requires_grad = not args.freeze_projector

    model_name = f"{args.base_model_id}_causal-lm{'_freeze-vision' if args.freeze_vision else ''}{'_freeze-projector' if args.freeze_projector else ''}_vsr"
    model_name += f"_b{args.train_batch_size}x{args.gradient_accumulation_steps}"
    model_name += f"_{args.learning_rate}_e{args.num_train_epochs}"
    model_name += f"_{args.dataset_type}-train-{num_train_data}"
    model_name += f"_lora-r{args.lora_r}-a{args.lora_alpha}"
    output_dir = f"modeling/{model_name}"

    # model.print_trainable_parameters()

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
        # use_cpu=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    def collate_fn(examples):
        # captions = ["answer en " + example["caption"] for example in examples]
        # labels= ["True" if example["label"] else "False" for example in examples]
        # images = [example["image"] for example in examples]

        # tokens = processor(
        #     text=captions, 
        #     images=images, 
        #     suffix=labels, 
        #     return_tensors="pt", 
        #     padding="longest"
        # )
        # tokens = tokens.to(device)
        # # return tokens

        max_length = 1024

        images = []
        chats = []
        for example in examples:
            images.append(example["image"])

            caption = example["caption"].strip()
            prompt = (
                "Look at the image and decide whether the statement is True or False.\n"
                f"Statement: {caption}\n"
                "Answer with exactly one word: True or False."
            )

            answer = "True" if example["label"] else "False"

            # Build a 2-turn conversation
            chats.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # placeholder; actual PIL image passed separately
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer},
                    ],
                },
            ])

        # 1) Full chat text (prompt + answer)
        full_texts = [
            processor.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,
            )
            for chat in chats
        ]
        # print('full_texts:', full_texts)

        # 2) Prompt-only chat text (prompt + assistant header)
        prompt_texts = [
            processor.apply_chat_template(
                chat[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            for chat in chats
        ]
        # print('prompt_texts:', prompt_texts)

        # 3) Tokenize full multimodal inputs (text + images)
        encoding = processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False,
        )

        tokenizer = processor.tokenizer
        pad_id = tokenizer.pad_token_id

        # 4) Tokenize prompt-only text (text only, cheaper) to compute prompt lengths
        prompt_ids = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,  # chat template already includes special tokens
        )["input_ids"]

        input_ids = encoding["input_ids"]

        # Mask prompt tokens (handle both left/right padding)
        prompt_lens = (prompt_ids != pad_id).sum(dim=1)

        labels = input_ids.clone()
        batch_size, seq_len = labels.shape

        # Mask padding positions
        labels[input_ids == pad_id] = -100

        for i in range(batch_size):
            prompt_len = int(prompt_lens[i].item())
            labels[i, min(prompt_len, seq_len)] = -100

        encoding["labels"] = labels
        return encoding

    trainer = Trainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_fn,
        callbacks=[AimCallback(experiment=model_name)] if args.log_with_aim else None,
        args=training_args
    )

    trainer.train()

'''
full_texts:
[
    '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Look at the image and decide whether the statement is True or False.\nStatement: The person is in the refrigerator.\nAnswer with exactly one word: True or False.<|im_end|>\n<|im_start|>assistant\nTrue<|im_end|>\n',
    '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Look at the image and decide whether the statement is True or False.\nStatement: The couch is beneath the cat.\nAnswer with exactly one word: True or False.<|im_end|>\n<|im_start|>assistant\nFalse<|im_end|>\n',
    '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Look at the image and decide whether the statement is True or False.\nStatement: The cat is in the suitcase.\nAnswer with exactly one word: True or False.<|im_end|>\n<|im_start|>assistant\nFalse<|im_end|>\n',
    '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Look at the image and decide whether the statement is True or False.\nStatement: The motorcycle is in front of the dog.\nAnswer with exactly one word: True or False.<|im_end|>\n<|im_start|>assistant\nTrue<|im_end|>\n'
]
prompt_texts:
[
    '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Look at the image and decide whether the statement is True or False.\nStatement: The person is in the refrigerator.\nAnswer with exactly one word: True or False.<|im_end|>\n<|im_start|>assistant\n',
    '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Look at the image and decide whether the statement is True or False.\nStatement: The couch is beneath the cat.\nAnswer with exactly one word: True or False.<|im_end|>\n<|im_start|>assistant\n',
    '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Look at the image and decide whether the statement is True or False.\nStatement: The cat is in the suitcase.\nAnswer with exactly one word: True or False.<|im_end|>\n<|im_start|>assistant\n',
    '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Look at the image and decide whether the statement is True or False.\nStatement: The motorcycle is in front of the dog.\nAnswer with exactly one word: True or False.<|im_end|>\n<|im_start|>assistant\n'
]

python scripts/finetuning_qwen3vl.py \
    --dataset_type "random" \
    --vsr_data_dir "data/vsr" \
    --freeze_vision

  Number of trainable parameters = 2,127,532,032
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB. GPU 0 has a total capacity of 39.38 GiB of which 27.38 MiB is free. Including non-PyTorch memory, this process has 39.35 GiB memory in use. Of the allocated memory 36.43 GiB is allocated by PyTorch, and 2.41 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

  Number of trainable parameters = 1,720,574,976
fine
'''