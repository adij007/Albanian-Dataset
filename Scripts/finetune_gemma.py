#!/usr/bin/env python3
"""
LoRA fine-tune Gemma 4 E4B on the Albanian Q&A dataset.
========================================================

Reads JSON/albanian_qa_dataset.json (instruction / input / output schema),
converts each row to a Gemma chat-template `messages` pair, and trains a
small LoRA adapter on top of the frozen base model.

Outputs (default):
  D:/Albanian-Dataset/checkpoints/gemma-albanian-lora/
    adapter_config.json, adapter_model.safetensors, tokenizer files

Examples
--------
  # quick smoke test (CPU, ~50 optimizer steps)
  python Scripts/finetune_gemma.py --max_steps 50

  # full single-epoch run
  python Scripts/finetune_gemma.py --epochs 1

  # custom data / output
  python Scripts/finetune_gemma.py `
      --data D:/Albanian-Dataset/JSON/albanian_qa_dataset.json `
      --output D:/Albanian-Dataset/checkpoints/run-01

Note: training a 5B-parameter model on CPU is extremely slow. Use the
`--max_steps` flag to validate the pipeline first; run the real training
on a GPU machine when one is available.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


REPO_ROOT     = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "D:/models/gemma-4-e4b-it"
DEFAULT_DATA  = REPO_ROOT / "JSON" / "albanian_qa_dataset.json"
DEFAULT_OUT   = REPO_ROOT / "checkpoints" / "gemma-albanian-lora"


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset(json_path: Path) -> Dataset:
    """Load the instruction/input/output JSON and convert to chat messages."""
    with open(json_path, encoding="utf-8") as f:
        rows = json.load(f)

    samples = []
    for ex in rows:
        instr = (ex.get("instruction") or "").strip()
        inp   = (ex.get("input")       or "").strip()
        out   = (ex.get("output")      or "").strip()
        if not instr or not out:
            continue
        user_text = f"{instr}\n\n{inp}" if inp else instr
        samples.append({
            "messages": [
                {"role": "user",      "content": user_text},
                {"role": "assistant", "content": out},
            ]
        })
    if not samples:
        raise RuntimeError(f"No valid rows in {json_path}")
    return Dataset.from_list(samples)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_base_model(path: str, dtype):
    """Gemma 4 is multimodal; prefer the ImageTextToText auto-class and fall
    back to CausalLM if the running transformers version doesn't expose it."""
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            path, torch_dtype=dtype, low_cpu_mem_usage=True,
        )
    except Exception:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, low_cpu_mem_usage=True,
        )


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def pick_dtype(device: str):
    if device in {"cuda", "xpu"}:
        return torch.float16
    return torch.float32


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--data",         default=str(DEFAULT_DATA))
    parser.add_argument("--output",       default=str(DEFAULT_OUT))
    parser.add_argument("--device",       choices=["auto", "cpu", "cuda", "xpu"],
                        default="auto")
    parser.add_argument("--epochs",       type=float, default=1.0)
    parser.add_argument("--max_steps",    type=int,   default=-1)
    parser.add_argument("--batch_size",   type=int,   default=1)
    parser.add_argument("--grad_accum",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--max_seq_len",  type=int,   default=512)
    parser.add_argument("--lora_r",       type=int,   default=8)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_ratio",   type=float, default=0.02)
    parser.add_argument("--logging_steps",type=int,   default=10)
    parser.add_argument("--save_steps",   type=int,   default=200)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = pick_dtype(device)
    has_accel = device in {"cuda", "xpu"}
    print(f"[finetune] device={device} | dtype={dtype}")
    if not has_accel:
        print("[finetune] WARNING: CPU training of a 5B model is extremely slow.")
        print("           Use --max_steps 50 (or similar) for a smoke test, and")
        print("           run the real training on a CUDA/XPU machine when available.")

    print(f"[finetune] Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[finetune] Loading base model from {args.model}")
    model = load_base_model(args.model, dtype)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print(f"[finetune] Loading dataset from {args.data}")
    full_ds = build_dataset(Path(args.data))
    split   = full_ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[finetune] train={len(train_ds):,} | eval={len(eval_ds):,}")

    sft_cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=max(args.save_steps, 50),
        bf16=False,
        fp16=has_accel,
        max_seq_length=args.max_seq_len,
        packing=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("[finetune] Starting training...")
    trainer.train()

    print(f"[finetune] Saving adapter to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("[finetune] DONE.")


if __name__ == "__main__":
    main()
