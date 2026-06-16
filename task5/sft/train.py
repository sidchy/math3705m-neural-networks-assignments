from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

from task5lib.io import read_jsonl, write_json
from task5lib.sft import render_prompt


def format_example(row: dict) -> dict:
    text = render_prompt(row["instruction"], row["input"]) + row["output"]
    return {"text": text}


def load_dataset(path: str) -> Dataset:
    rows = [format_example(row) for row in read_jsonl(path)]
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", default="runs/sft/adapter_model")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--smoke_steps", type=int, default=0)
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    train_dataset = load_dataset(args.data)
    eval_dataset = load_dataset(args.eval)
    training_args = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        optim="adamw_8bit",
        fp16=True,
        bf16=False,
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        max_steps=args.smoke_steps if args.smoke_steps > 0 else -1,
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset, args=training_args)
    result = trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    write_json(Path(args.out).parent / "training_args.json", vars(args))
    write_json(Path(args.out).parent / "metrics.json", result.metrics)


if __name__ == "__main__":
    main()
