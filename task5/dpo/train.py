from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset

from task5lib.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_or_adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", default="runs/dpo/adapter_model")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()

    from unsloth import FastLanguageModel, PatchDPOTrainer
    from trl import DPOTrainer, DPOConfig

    PatchDPOTrainer()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_or_adapter,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=False,
    )
    train_dataset = Dataset.from_list(read_jsonl(args.data))
    eval_dataset = Dataset.from_list(read_jsonl(args.eval))
    config = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        optim="adamw_8bit",
        fp16=True,
        bf16=False,
        logging_steps=10,
        save_steps=200,
        max_steps=args.max_steps,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    result = trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    write_json(Path(args.out).parent / "training_args.json", vars(args))
    write_json(Path(args.out).parent / "metrics.json", result.metrics)


if __name__ == "__main__":
    main()
