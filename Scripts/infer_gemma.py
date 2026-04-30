#!/usr/bin/env python3
"""
Interactive Albanian Q&A chat with Gemma 4 E4B (+ optional LoRA adapter).
========================================================================

Loads the base model from D:/models/gemma-4-e4b-it and, if available,
applies the LoRA adapter trained by Scripts/finetune_gemma.py.

Examples
--------
  # use adapter from default location
  python Scripts/infer_gemma.py

  # base model only
  python Scripts/infer_gemma.py --no_adapter

  # custom adapter
  python Scripts/infer_gemma.py --adapter D:/Albanian-Dataset/checkpoints/run-01

  # one-shot prompt (no interactive loop)
  python Scripts/infer_gemma.py --prompt "Cilat janë sinonimet e fjalës 'shtëpi'?"
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_MODEL   = "D:/models/gemma-4-e4b-it"
DEFAULT_ADAPTER = REPO_ROOT / "checkpoints" / "gemma-albanian-lora"


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


def load_base_model(path: str, dtype):
    """Prefer ImageTextToText (Gemma 4 multimodal); fall back to CausalLM."""
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


def generate(model, tokenizer, history, device, args) -> str:
    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    # Run batched generation to better saturate GPU/XPU compute.
    batched_prompts = [prompt] * max(1, args.gpu_batch_size)
    inputs = tokenizer(
        batched_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _sync_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    if device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def _clear_cache(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg


def autotune_gpu_batch(model, tokenizer, device: str, args) -> int:
    if device not in {"cuda", "xpu"}:
        print("[autotune] Skipped: autotune only applies to cuda/xpu devices.")
        return 1

    probe_history = [{"role": "user", "content": "Përshëndetje! Jep një përgjigje të shkurtër."}]
    best = 1
    batch = max(1, args.gpu_batch_size)
    ceiling = max(batch, args.autotune_max_batch)

    print(f"[autotune] Probing batch sizes up to {ceiling}...")
    while batch <= ceiling:
        args.gpu_batch_size = batch
        try:
            t0 = time.perf_counter()
            _ = generate(model, tokenizer, probe_history, device, args)
            _sync_device(device)
            dt = time.perf_counter() - t0
            best = batch
            print(f"[autotune] batch={batch:>2} OK   ({dt:.2f}s)")
            batch += args.autotune_step
        except RuntimeError as exc:
            if _is_oom_error(exc):
                print(f"[autotune] batch={batch:>2} OOM  -> stopping.")
                _clear_cache(device)
                break
            raise

    args.gpu_batch_size = best
    print(f"[autotune] Best stable --gpu_batch_size: {best}")
    print(
        "[autotune] Suggested command: "
        f"python Scripts/infer_gemma.py --device {device} "
        f"--gpu_batch_size {best} --max_input_tokens {args.max_input_tokens} "
        f"--max_new_tokens {args.max_new_tokens} --max_history_turns {args.max_history_turns}"
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--adapter",        default=str(DEFAULT_ADAPTER))
    parser.add_argument("--no_adapter",     action="store_true",
                        help="Skip loading the LoRA adapter (run base model).")
    parser.add_argument("--prompt",         default=None,
                        help="Run a single prompt and exit (non-interactive).")
    parser.add_argument("--device",         choices=["auto", "cpu", "cuda", "xpu"],
                        default="auto")
    parser.add_argument("--max_new_tokens", type=int,   default=128)
    parser.add_argument("--max_input_tokens", type=int, default=768)
    parser.add_argument("--max_history_turns", type=int, default=6)
    parser.add_argument("--gpu_batch_size", type=int, default=1,
                        help="Duplicate the same prompt N times to increase GPU/XPU utilization.")
    parser.add_argument("--autotune_batch", action="store_true",
                        help="Probe the highest stable gpu_batch_size for current token settings.")
    parser.add_argument("--autotune_max_batch", type=int, default=32,
                        help="Upper bound for --autotune_batch search.")
    parser.add_argument("--autotune_step", type=int, default=1,
                        help="Step size when probing gpu_batch_size.")
    parser.add_argument("--temperature",    type=float, default=0.2)
    parser.add_argument("--top_p",          type=float, default=0.95)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = pick_dtype(device)
    print(f"[infer] device={device} | dtype={dtype}")

    print(f"[infer] Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"[infer] Loading base model from {args.model}")
    model = load_base_model(args.model, dtype)

    adapter_path = Path(args.adapter)
    if not args.no_adapter and adapter_path.exists():
        print(f"[infer] Loading LoRA adapter from {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
    else:
        if args.no_adapter:
            print("[infer] --no_adapter set: running base model only.")
        else:
            print(f"[infer] Adapter not found at {adapter_path}: running base model.")

    model.to(device).eval()

    if args.autotune_batch:
        autotune_gpu_batch(model, tokenizer, device, args)
        return

    if args.prompt:
        history = [{"role": "user", "content": args.prompt}]
        reply = generate(model, tokenizer, history, device, args)
        print()
        print(reply)
        return

    print()
    print("Bisedo me modelin (shkruaj 'exit' ose 'quit' për të dalë).")
    print("=" * 60)

    history: list[dict] = []
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        if len(history) > (args.max_history_turns * 2):
            history = history[-(args.max_history_turns * 2):]

        history.append({"role": "user", "content": user})
        reply = generate(model, tokenizer, history, device, args)
        history.append({"role": "assistant", "content": reply})
        print(f"\n{reply}\n")


if __name__ == "__main__":
    main()
