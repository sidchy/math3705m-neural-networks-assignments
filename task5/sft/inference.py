from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from task5lib.io import read_jsonl
from task5lib.sft import render_prompt


def generate_one(model, tokenizer, instruction: str, input_text: str, max_new_tokens: int = 128) -> str:
    prompt = render_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("### 回答")[-1].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_or_adapter", default="")
    parser.add_argument("--base_model", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=False,
    )
    if args.model_or_adapter:
        model.load_adapter(args.model_or_adapter)

    test_rows = read_jsonl(args.test)
    wz_rows = [row for row in test_rows if row["task_type"] == "wz_to_zh"]
    if args.limit > 0:
        wz_rows = wz_rows[:args.limit]

    predictions = []
    for row in wz_rows:
        pred = generate_one(model, tokenizer, row["instruction"], row["input"])
        predictions.append({
            "id": row["id"],
            "input": row["input"],
            "reference": row["output"],
            "prediction": pred,
            "task_type": row["task_type"],
            "model": args.model_or_adapter or args.base_model,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"wrote {len(predictions)} predictions to {args.out}")


if __name__ == "__main__":
    main()
