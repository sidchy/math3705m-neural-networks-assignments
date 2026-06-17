from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader, Dataset
from task5lib.io import read_jsonl, write_json


class DPODataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
        self.rows = read_jsonl(path)
        self.tok = tokenizer
        self.max_len = max_length
        self.ref_chosen = None  # set by precompute_ref_logps
        self.ref_rejected = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt = row["prompt"]
        chosen_ids = self.tok.encode(prompt, row["chosen"], truncation=True, max_length=self.max_len)
        rejected_ids = self.tok.encode(prompt, row["rejected"], truncation=True, max_length=self.max_len)
        item = {
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
        }
        if self.ref_chosen is not None:
            item["ref_chosen"] = self.ref_chosen[idx]
            item["ref_rejected"] = self.ref_rejected[idx]
        return item


def collate_fn(batch):
    max_len = 0
    for item in batch:
        max_len = max(max_len, len(item["chosen_ids"]), len(item["rejected_ids"]))

    chosen_ids_list, chosen_mask_list = [], []
    rejected_ids_list, rejected_mask_list = [], []
    ref_chosen_list, ref_rejected_list = [], []

    for item in batch:
        for src, ids_list, mask_list in [
            ("chosen_ids", chosen_ids_list, chosen_mask_list),
            ("rejected_ids", rejected_ids_list, rejected_mask_list),
        ]:
            ids = item[src]
            pad_len = max_len - len(ids)
            ids_list.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)]))
            mask_list.append(torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))

        if "ref_chosen" in item:
            ref_chosen_list.append(item["ref_chosen"])
            ref_rejected_list.append(item["ref_rejected"])

    result = {
        "chosen_ids": torch.stack(chosen_ids_list),
        "chosen_mask": torch.stack(chosen_mask_list),
        "rejected_ids": torch.stack(rejected_ids_list),
        "rejected_mask": torch.stack(rejected_mask_list),
    }
    if ref_chosen_list:
        result["ref_chosen"] = torch.stack(ref_chosen_list)
        result["ref_rejected"] = torch.stack(ref_rejected_list)
    return result


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = policy_logratios - ref_logratios
    logits = torch.clamp(logits, -50, 50)
    return -torch.nn.functional.logsigmoid(beta * logits).mean()


def compute_logps(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    # Always do log_softmax in fp32. On T4 the model may return fp16 logits even
    # when the base weights are fp32, and fp16 log_softmax is fragile here.
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * shift_mask
    return token_logps.sum(dim=-1)


def finite_report(model, trainable_only=True):
    bad = []
    dtype_counts = Counter()
    max_abs = 0.0
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        dtype_counts[str(param.dtype)] += param.numel()
        if not torch.isfinite(param).all():
            bad.append(name)
        if param.numel():
            max_abs = max(max_abs, float(param.detach().abs().max().float().cpu()))
    return bad, dtype_counts, max_abs


def grad_report(model):
    bad = []
    max_abs = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            bad.append(name)
        if param.grad.numel():
            max_abs = max(max_abs, float(param.grad.detach().abs().max().float().cpu()))
    return bad, max_abs


def should_train_lora(name: str, scope: str) -> bool:
    if "lora" not in name:
        return False
    if scope == "all":
        return True
    if scope == "attention":
        return any(part in name for part in ("q_proj", "k_proj", "v_proj", "o_proj"))
    if scope == "mlp":
        return any(part in name for part in ("gate_proj", "up_proj", "down_proj"))
    raise ValueError(f"Unknown LoRA scope: {scope}")


@torch.no_grad()
def precompute_ref_logps(model, dataset, batch_size, log_every=20):
    """Compute reference logprobs for every item, store in dataset."""
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_chosen = []
    all_rejected = []
    total = len(dl)
    for i, batch in enumerate(dl):
        chosen_ids = batch["chosen_ids"].cuda()
        chosen_mask = batch["chosen_mask"].cuda()
        rejected_ids = batch["rejected_ids"].cuda()
        rejected_mask = batch["rejected_mask"].cuda()
        both_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        both_mask = torch.cat([chosen_mask, rejected_mask], dim=0)
        both_logps = compute_logps(model, both_ids, both_mask).cpu()
        chosen_logps, rejected_logps = both_logps.chunk(2, dim=0)
        all_chosen.append(chosen_logps)
        all_rejected.append(rejected_logps)
        if log_every > 0 and (i == 0 or (i + 1) % log_every == 0 or i + 1 == total):
            print(f"ref precompute {i + 1}/{total}", flush=True)
    dataset.ref_chosen = torch.cat(all_chosen)
    dataset.ref_rejected = torch.cat(all_rejected)
    return dataset.ref_chosen, dataset.ref_rejected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/root/.cache/modelscope/unsloth/Qwen3.5-2B")
    p.add_argument("--adapter", default="runs/sft/adapter_model")
    p.add_argument("--data", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--out", default="runs/dpo/adapter_model")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--precompute_bs", type=int, default=16)
    p.add_argument("--precompute_log_every", type=int, default=20)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--adam_eps", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loader", choices=["unsloth", "native"], default="unsloth",
                   help="Model loader. native uses transformers+PEFT and avoids Unsloth patched backward.")
    p.add_argument("--lora_scope", choices=["attention", "mlp", "all"], default="attention",
                   help="Which LoRA tensors to train. Default attention avoids unstable Unsloth MLP LoRA backward on T4.")
    p.add_argument("--train_mode", action="store_true",
                   help="Use model.train() during DPO. Default keeps model.eval() to avoid Unsloth train-time fp16 paths.")
    args = p.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Single model: base + LoRA adapter. `native` is slower but avoids Unsloth's
    # patched backward path, which produced non-finite LoRA gradients on T4.
    if args.loader == "unsloth":
        from unsloth import FastLanguageModel
        model, _ = FastLanguageModel.from_pretrained(
            args.base,
            max_seq_length=args.max_length,
            dtype=torch.float32,
            load_in_4bit=False,
        )
    else:
        print("Loading base model with native transformers+PEFT in fp32...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if hasattr(model, "config"):
            model.config.use_cache = False
        model = model.cuda()
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True).cuda()

    # Only train selected LoRA params and force every trainable tensor to fp32. Do this
    # after PeftModel.from_pretrained because PEFT may create adapter tensors
    # after Unsloth has selected its "16bit LoRA" path.
    for n, p in model.named_parameters():
        if "lora" in n:
            p.data = p.data.float()
            p.requires_grad = should_train_lora(n, args.lora_scope)
        else:
            p.requires_grad = False

    if hasattr(model, "config"):
        model.config.use_cache = False

    bad, dtype_counts, max_abs = finite_report(model, trainable_only=True)
    print(f"Trainable dtype counts before ref precompute: {dict(dtype_counts)} max_abs={max_abs:.4e}")
    if bad:
        raise RuntimeError(f"Non-finite trainable params before training: {bad[:5]}")
    model.eval()

    train_ds = DPODataset(args.data, tokenizer, args.max_length)
    eval_ds = DPODataset(args.eval, tokenizer, args.max_length)

    # Precompute reference logprobs (SFT model = initial policy before training)
    print("Precomputing reference logprobs on train set...")
    train_chosen, train_rejected = precompute_ref_logps(model, train_ds, args.precompute_bs, args.precompute_log_every)
    print(f"Done. train: {train_chosen.shape[0]} items, "
          f"chosen mean={train_chosen.float().mean():.2f} "
          f"rejected mean={train_rejected.float().mean():.2f}")

    print("Precomputing reference logprobs on eval set...")
    eval_chosen, eval_rejected = precompute_ref_logps(model, eval_ds, args.precompute_bs, args.precompute_log_every)
    print(f"Done. eval: {eval_chosen.shape[0]} items, "
          f"chosen mean={eval_chosen.float().mean():.2f} "
          f"rejected mean={eval_rejected.float().mean():.2f}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable ({args.lora_scope} LoRA): {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print("Trainable module counts:", dict(Counter(
        "attention" if any(part in n for part in ("q_proj", "k_proj", "v_proj", "o_proj")) else "mlp"
        for n, p in model.named_parameters() if p.requires_grad
    )))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, eps=args.adam_eps, foreach=False)

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    metrics_log = []
    global_step = 0

    for epoch in range(args.epochs):
        if args.train_mode:
            model.train()
        else:
            # DPO normally disables dropout. Keeping eval() also avoids Unsloth
            # train-time fast paths; gradients still flow to LoRA parameters.
            model.eval()
        epoch_losses = []
        optimizer.zero_grad()

        for step, batch in enumerate(dl):
            chosen_ids = batch["chosen_ids"].cuda()
            chosen_mask = batch["chosen_mask"].cuda()
            rejected_ids = batch["rejected_ids"].cuda()
            rejected_mask = batch["rejected_mask"].cuda()

            p_chosen = compute_logps(model, chosen_ids, chosen_mask)
            p_rejected = compute_logps(model, rejected_ids, rejected_mask)
            r_chosen = batch["ref_chosen"].cuda()
            r_rejected = batch["ref_rejected"].cuda()

            loss = dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected, args.beta) / args.grad_accum
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"epoch {epoch+1} step {step}: loss={loss.item()}, "
                      f"p_chosen={p_chosen.float().mean().item():.2f} "
                      f"p_rejected={p_rejected.float().mean().item():.2f} "
                      f"r_chosen={r_chosen.float().mean().item():.2f} "
                      f"r_rejected={r_rejected.float().mean().item():.2f}")
                optimizer.zero_grad()
                continue
            loss.backward()
            bad_grads, grad_max = grad_report(model)
            if bad_grads:
                raise RuntimeError(
                    f"Non-finite gradients immediately after backward at epoch {epoch+1} step {step}: "
                    f"{bad_grads[:8]} grad_max={grad_max:.4e} "
                    f"p_chosen={p_chosen.float().mean().item():.2f} "
                    f"p_rejected={p_rejected.float().mean().item():.2f} "
                    f"r_chosen={r_chosen.float().mean().item():.2f} "
                    f"r_rejected={r_rejected.float().mean().item():.2f}"
                )
            epoch_losses.append(loss.item() * args.grad_accum)

            if (step + 1) % args.grad_accum == 0:
                bad_grads, grad_max = grad_report(model)
                if bad_grads:
                    raise RuntimeError(f"Non-finite gradients before optimizer.step at step {step}: {bad_grads[:5]}")
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"Non-finite grad norm before optimizer.step at step {step}: {grad_norm}")
                optimizer.step()
                bad_params, dtype_counts, param_max = finite_report(model, trainable_only=True)
                if bad_params:
                    raise RuntimeError(
                        f"Non-finite trainable params after optimizer.step at step {step}: "
                        f"{bad_params[:5]} dtype_counts={dict(dtype_counts)} grad_max={grad_max:.4e} param_max={param_max:.4e}"
                    )
                optimizer.zero_grad()
                global_step += 1

            if step % 10 == 0:
                avg = sum(epoch_losses[-50:]) / len(epoch_losses[-50:])
                print(f"epoch {epoch+1} step {step} loss={avg:.4f}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))

        model.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in eval_dl:
                chosen_ids = batch["chosen_ids"].cuda()
                chosen_mask = batch["chosen_mask"].cuda()
                rejected_ids = batch["rejected_ids"].cuda()
                rejected_mask = batch["rejected_mask"].cuda()

                p_chosen = compute_logps(model, chosen_ids, chosen_mask)
                p_rejected = compute_logps(model, rejected_ids, rejected_mask)
                r_chosen = batch["ref_chosen"].cuda()
                r_rejected = batch["ref_rejected"].cuda()

                eval_losses.append(dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected, args.beta).item())

        eval_loss = sum(eval_losses) / max(1, len(eval_losses))
        metrics_log.append({"epoch": epoch + 1, "train_loss": train_loss, "eval_loss": eval_loss})
        print(f"epoch {epoch+1}: train_loss={train_loss:.4f} eval_loss={eval_loss:.4f}")

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    write_json(out_dir.parent / "dpo_metrics.json", metrics_log)
    write_json(out_dir.parent / "dpo_args.json", vars(args))
    print(f"done. saved to {args.out}")


if __name__ == "__main__":
    main()
