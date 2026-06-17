from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from task5lib.io import read_jsonl
from task5lib.sft import render_prompt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="")
    p.add_argument("--base", default="/root/.cache/modelscope/unsloth/Qwen3.5-2B")
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--loader", choices=["unsloth", "native"], default="unsloth",
                   help="unsloth for SFT adapters, native for DPO (fp32) adapters")
    a = p.parse_args()

    if a.loader == "native":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(a.adapter or a.base, trust_remote_code=True)
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32, trust_remote_code=True).cuda()
        if a.adapter:
            model = PeftModel.from_pretrained(model, a.adapter, is_trainable=False).cuda()
    else:
        from unsloth import FastLanguageModel
        from peft import PeftModel
        model, tok = FastLanguageModel.from_pretrained(
            a.base, max_seq_length=256, dtype=None, load_in_4bit=False
        )
        tok.pad_token = tok.eos_token
        if a.adapter:
            model = PeftModel.from_pretrained(model, a.adapter, is_trainable=False).to("cuda")

    model.eval()

    rows = [r for r in read_jsonl(a.test) if r["task_type"] == "wz_to_zh"]
    if a.limit:
        rows = rows[:a.limit]
    preds = []
    for i in range(0, len(rows), a.bs):
        batch = rows[i:i + a.bs]
        prompts = [render_prompt(r["instruction"], r["input"]) for r in batch]
        inp = tok(text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        out = model.generate(**inp, max_new_tokens=128, do_sample=False, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
        for j, r in enumerate(batch):
            text = tok.decode(out[j], skip_special_tokens=True)
            pred = text.split("### 回答")[-1].strip().split("。")[0].strip()
            print(f"[{i+j+1}/{len(rows)}] {pred[:60]}")
            preds.append({"id": r["id"], "input": r["input"], "reference": r["output"], "prediction": pred, "task_type": r["task_type"], "model": a.adapter or a.base})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in preds) + "\n")
    print(f"wrote {len(preds)}")

if __name__ == "__main__":
    main()
