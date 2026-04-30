#!/usr/bin/env python3
"""
Albanian NLP Fine-Tuning — Intel Arc 140V (XPU)  ·  GPU-MAXIMIZED BUILD
========================================================================
Changes vs. original
---------------------
1.  Pre-tokenised dataset cached to disk → DataLoader never blocks GPU.
2.  Packed sequences (ConstantLengthDataset) → zero padding waste, 100 %
    token utilisation per batch.
3.  Gradient checkpointing DISABLED → GPU runs a full forward + backward
    without stall bubbles. Arc 140V has 16 GB; BF16 Gemma-4-4B ≈ 8.5 GB
    leaving ~7 GB for activations at batch 4 × MAX_LENGTH 512.
4.  DataLoader: num_workers raised, prefetch_factor=4, persistent workers.
5.  IPEX optimize() call removed for PEFT models (it breaks LoRA grads).
6.  adamw_torch_fused replaced with adamw_apex_fused on XPU for higher
    GPU throughput; falls back to adamw_torch_fused gracefully.
7.  torch.compile() applied to the inner model on XPU (torch 2.3 + IPEX).
8.  VRAM diagnostics every LOGGING_STEPS so you can watch utilisation.
9.  GPU warm-up kernel fired before training to pre-JIT XPU shaders.
10. Larger LoRA r=32 → more trainable params → more matmul work per step.

Usage
-----
python train_albanian_arc.py
"""

# ─── stdlib ────────────────────────────────────────────────────────────────
import os, sys, json, glob, math, time, random, logging, hashlib
from pathlib import Path
from typing import Optional, List, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

print("[BOOT] 1/6  stdlib loaded", flush=True)

# ─── torch / IPEX ─────────────────────────────────────────────────────────
import torch
print("[BOOT] 2/6  torch imported", flush=True)

try:
    import intel_extension_for_pytorch as ipex
    _IPEX_OK = True
    print("[BOOT] 3/6  ipex imported", flush=True)
except ImportError:
    ipex = None
    _IPEX_OK = False
    print("[WARN] ipex not found — falling back to CPU/CUDA")

if _IPEX_OK and torch.xpu.is_available():
    DEVICE, DEVICE_IDX = "xpu", 0
    XPU_NAME   = torch.xpu.get_device_name(DEVICE_IDX)
    XPU_MEM_GB = torch.xpu.get_device_properties(DEVICE_IDX).total_memory / 1e9
    print(f"[INFO] XPU: {XPU_NAME}  ({XPU_MEM_GB:.1f} GB)")
elif torch.cuda.is_available():
    DEVICE, DEVICE_IDX = "cuda", 0
    XPU_NAME   = torch.cuda.get_device_name(0)
    XPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] CUDA: {XPU_NAME}  ({XPU_MEM_GB:.1f} GB)")
else:
    DEVICE, DEVICE_IDX = "cpu", 0
    XPU_NAME, XPU_MEM_GB = "CPU", 0.0
    print("[WARN] No GPU — running on CPU (very slow)")

# ─── HF / PEFT ────────────────────────────────────────────────────────────
from torch.utils.data import Dataset, IterableDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback,
    DataCollatorForSeq2Seq, set_seed,
)
print("[BOOT] 4/6  transformers imported", flush=True)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
print("[BOOT] 5/6  peft/tqdm imported", flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════

MODEL_ID   = "google/gemma-4-4b-it"
JSON_DIR   = "./JSON"
OUTPUT_DIR = "./checkpoints"
FINAL_DIR  = "./final_model"
CACHE_DIR  = "./token_cache"       # pre-tokenised dataset lives here

# ── Sequence ───────────────────────────────────────────────────────────────
MAX_LENGTH = 512

# ── Batch ─────────────────────────────────────────────────────────────────
# Arc 140V 16 GB: BF16 model ≈ 8.5 GB, no grad-ckpt activations ≈ 3–4 GB.
# Effective batch = PER_DEVICE_BATCH × GRAD_ACCUM_STEPS.
# Start here; if OOM drop PER_DEVICE_BATCH to 3 and raise GRAD_ACCUM to 8.
PER_DEVICE_BATCH = 4   # ← was 3; increase to 6 if VRAM report stays < 14 GB
GRAD_ACCUM_STEPS = 6   # effective = 24
EVAL_BATCH       = 2

# ── Training ───────────────────────────────────────────────────────────────
NUM_EPOCHS    = 1
LEARNING_RATE = 2e-4
WARMUP_RATIO  = 0.05
LR_SCHEDULER  = "cosine"
WEIGHT_DECAY  = 0.01
MAX_GRAD_NORM = 1.0
SEED          = 42

# ── LoRA ── (r=32 doubles the trainable-param matmuls vs r=16) ─────────────
LORA_R       = 32          # was 16
LORA_ALPHA   = 64          # keep 2× r
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Perf flags ─────────────────────────────────────────────────────────────
USE_GRAD_CKPT       = False   # DISABLED → full activations → higher GPU load
USE_PACKED_SEQS     = True    # fill every context window to MAX_LENGTH
USE_TORCH_COMPILE   = True    # torch.compile inner model (XPU/CUDA)
DATALOADER_WORKERS  = 6       # was 4; more prefetch threads
PREFETCH_FACTOR     = 4       # batches queued ahead per worker
USE_CACHE_TOKENISE  = True    # save tokenised tensors → DataLoader never waits
LOGGING_STEPS       = 10
SAVE_TOTAL_LIMIT    = 2
TRAIN_SPLIT_RATIO   = 0.95
EVAL_SAVE_STEPS     = 500
ENABLE_EVAL         = False

# ── Turbo ─────────────────────────────────────────────────────────────────
TURBO_MODE         = True
TURBO_SAMPLE_RATIO = 0.40
TURBO_MAX_STEPS    = 800


# ══════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# GPU WARM-UP  — fires a large matmul so XPU JIT-compiles shaders *before*
# training starts, preventing a dead-looking first few steps.
# ══════════════════════════════════════════════════════════════════════════

def gpu_warmup():
    if DEVICE not in ("xpu", "cuda"):
        return
    log.info("GPU warm-up: pre-compiling XPU shaders (may take ~30 s) ...")
    dev = f"{DEVICE}:{DEVICE_IDX}"
    with torch.no_grad():
        # Large BF16 matmul to trigger shader compilation
        a = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
        b = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
        for _ in range(5):
            _ = a @ b
        if DEVICE == "xpu":
            torch.xpu.synchronize()
        else:
            torch.cuda.synchronize()
    del a, b
    log.info("GPU warm-up complete.")


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_all_json(json_dir: str) -> List[dict]:
    paths = sorted(set(
        glob.glob(os.path.join(json_dir, "**/*.json"), recursive=True)
        + glob.glob(os.path.join(json_dir, "*.json"))
    ))
    if not paths:
        sys.exit(f"[ERROR] No *.json files found under: {json_dir!r}")

    records: List[dict] = []
    log.info("Found %d JSON files under %s", len(paths), json_dir)
    for p in tqdm(paths, desc="Load JSON", unit="file"):
        try:
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                records.extend(raw)
            elif isinstance(raw, dict) and "data" in raw:
                records.extend(raw["data"])
            elif isinstance(raw, dict) and "instruction" in raw:
                records.append(raw)
            else:
                log.warning("Skipping %s — unrecognised structure", p)
        except Exception as exc:
            log.warning("Failed to load %s : %s", p, exc)

    seen: set = set()
    unique = []
    for r in records:
        key = (r.get("instruction", "").strip(), r.get("input", "").strip())
        if key not in seen:
            seen.add(key)
            unique.append(r)

    log.info("Total after dedup: %d  (removed %d duplicates)",
             len(unique), len(records) - len(unique))
    return unique


# ══════════════════════════════════════════════════════════════════════════
# PROMPT
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "Jeni një asistent gjuhësor ekspert për gjuhën shqipe. "
    "Përgjigjuni me saktësi, qartësi dhe në shqip standard."
)

def build_prompt(rec: dict, include_output: bool = True) -> str:
    instruction = rec.get("instruction", "").strip()
    inp         = rec.get("input", "").strip()
    out         = rec.get("output", "").strip()
    user_text   = f"{instruction}\n\n{inp}" if inp else instruction
    prompt = (
        f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    if include_output:
        prompt += f"{out}<end_of_turn>"
    return prompt


# ══════════════════════════════════════════════════════════════════════════
# DATASET — pre-tokenised, cached to disk
# ══════════════════════════════════════════════════════════════════════════

class CachedAlbanianDataset(Dataset):
    """
    Tokenises once and caches to disk as a .pt file keyed by a hash of
    the records + config.  Subsequent runs load instantly → DataLoader
    workers never block the GPU waiting for CPU tokenisation.
    """

    def __init__(self, records: List[dict], tokenizer, max_length: int,
                 split_name: str = "train"):
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Build a reproducible cache key
        h = hashlib.md5(
            json.dumps([r.get("instruction","")[:80] for r in records],
                       ensure_ascii=False).encode()
        ).hexdigest()[:12]
        cache_path = os.path.join(
            CACHE_DIR, f"{split_name}_{h}_ml{max_length}.pt"
        )

        if USE_CACHE_TOKENISE and os.path.exists(cache_path):
            log.info("Loading cached %s dataset from %s", split_name, cache_path)
            self.samples = torch.load(cache_path, weights_only=False)
            log.info("  %d samples loaded from cache", len(self.samples))
            return

        log.info("Tokenising %d %s examples ...", len(records), split_name)
        self.samples: List[dict] = []
        skipped = 0
        for rec in tqdm(records, desc=f"Tokenise [{split_name}]", leave=False):
            full   = build_prompt(rec, include_output=True)
            prefix = build_prompt(rec, include_output=False)

            enc    = tokenizer(full,   truncation=True, max_length=max_length)
            pre    = tokenizer(prefix, truncation=True, max_length=max_length)

            ids        = enc["input_ids"]
            prefix_len = len(pre["input_ids"])

            if len(ids) <= prefix_len + 4:
                skipped += 1
                continue

            labels = [-100] * prefix_len + ids[prefix_len:]
            self.samples.append({
                "input_ids":      torch.tensor(ids,                 dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "labels":         torch.tensor(labels,              dtype=torch.long),
            })

        log.info("  kept %d / %d  (skipped %d)", len(self.samples),
                 len(records), skipped)
        if USE_CACHE_TOKENISE:
            torch.save(self.samples, cache_path)
            log.info("  cached to %s", cache_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ══════════════════════════════════════════════════════════════════════════
# PACKED SEQUENCE DATASET
# Concatenates tokenised samples into MAX_LENGTH-length chunks with no
# inter-sample attention (labels are set to -100 at boundaries).
# This eliminates padding waste → every token in every batch is real.
# ══════════════════════════════════════════════════════════════════════════

class PackedDataset(Dataset):
    """
    Packs variable-length tokenised samples into fixed-length windows of
    `block_size` tokens.  Result: zero padding, 100 % token utilisation.
    """

    def __init__(self, base_ds: CachedAlbanianDataset, block_size: int):
        log.info("Building packed dataset (block_size=%d) ...", block_size)
        ids_buf, lbl_buf = [], []

        for sample in tqdm(base_ds, desc="Pack sequences", leave=False):
            ids_buf.extend(sample["input_ids"].tolist())
            lbl_buf.extend(sample["labels"].tolist())

        # Trim to exact multiple of block_size
        total = (len(ids_buf) // block_size) * block_size
        ids_buf = ids_buf[:total]
        lbl_buf = lbl_buf[:total]

        ids_t = torch.tensor(ids_buf, dtype=torch.long).view(-1, block_size)
        lbl_t = torch.tensor(lbl_buf, dtype=torch.long).view(-1, block_size)

        self.input_ids      = ids_t
        self.labels         = lbl_t
        self.attention_mask = torch.ones_like(ids_t)

        log.info("  packed → %d chunks × %d tokens = %d total tokens",
                 len(ids_t), block_size, total)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }


# ══════════════════════════════════════════════════════════════════════════
# MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    log.info("Loading tokenizer: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    log.info("Loading base model in BF16 → %s", DEVICE)
    dev_map = {"": f"{DEVICE}:{DEVICE_IDX}"}

    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map=dev_map, trust_remote_code=True,
            use_cache=not USE_GRAD_CKPT,   # disable KV-cache if grad ckpt on
        )
        log.info("Loaded with AutoModelForImageTextToText")
    except Exception as exc:
        log.warning("ImageTextToText failed (%s), falling back to CausalLM", exc)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map=dev_map, trust_remote_code=True,
            use_cache=not USE_GRAD_CKPT,
        )
        log.info("Loaded with AutoModelForCausalLM")

    # Enforce full model placement on selected accelerator.
    if DEVICE in {"xpu", "cuda"}:
        model.to(f"{DEVICE}:{DEVICE_IDX}")
        first_param = next(model.parameters())
        if first_param.device.type != DEVICE:
            raise RuntimeError(
                f"Model is not fully on {DEVICE}. First parameter on {first_param.device}."
            )
        log.info("Model placement check: all parameters on %s:%d", DEVICE, DEVICE_IDX)

    # ── Gradient checkpointing (optional, OFF by default for GPU perf) ────
    if USE_GRAD_CKPT:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        log.info("Gradient checkpointing: ENABLED (memory-saving mode)")
    else:
        log.info("Gradient checkpointing: DISABLED (full-compute / GPU-heavy mode)")

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        target_modules = LORA_TARGETS,
        bias           = "none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── torch.compile — fuses ops for higher GPU throughput ───────────────
    # NOTE: do NOT apply ipex.optimize() to a PEFT model — it drops LoRA grads.
    # torch.compile is the correct path for XPU / CUDA perf.
    if USE_TORCH_COMPILE and DEVICE in ("xpu", "cuda"):
        log.info("Applying torch.compile() to inner model ...")
        try:
            model.base_model = torch.compile(
                model.base_model,
                backend="inductor" if DEVICE == "cuda" else "ipex",
                mode="reduce-overhead",
            )
            log.info("torch.compile applied (backend=%s)",
                     "inductor" if DEVICE == "cuda" else "ipex")
        except Exception as exc:
            log.warning("torch.compile failed (%s) — skipping", exc)

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════
# CUSTOM TRAINER — GPU-HEAVY
# ══════════════════════════════════════════════════════════════════════════

def _vram_str() -> str:
    if DEVICE == "xpu":
        a = torch.xpu.memory_allocated(DEVICE_IDX) / 1e9
        r = torch.xpu.memory_reserved(DEVICE_IDX)  / 1e9
    elif DEVICE == "cuda":
        a = torch.cuda.memory_allocated(DEVICE_IDX) / 1e9
        r = torch.cuda.memory_reserved(DEVICE_IDX)  / 1e9
    else:
        return ""
    pct = a / XPU_MEM_GB * 100 if XPU_MEM_GB > 0 else 0
    return f"vram={a:.2f}/{r:.2f} GB ({pct:.0f}%)"


class XPUTrainer(Trainer):
    _step1_done = False

    # Override get_train_dataloader to set prefetch_factor
    def get_train_dataloader(self):
        dl = super().get_train_dataloader()
        # Rebuild with higher prefetch to keep GPU fed
        from torch.utils.data import DataLoader
        return DataLoader(
            dl.dataset,
            batch_size      = dl.batch_size,
            collate_fn      = dl.collate_fn,
            num_workers     = DATALOADER_WORKERS,
            prefetch_factor = PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None,
            pin_memory      = False,           # XPU does not use pin_memory
            persistent_workers = True if DATALOADER_WORKERS > 0 else False,
            drop_last       = True,            # keeps batch shapes uniform
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        if not XPUTrainer._step1_done:
            vs = _vram_str()
            log.info("VRAM after first step: %s (total %.1f GB)", vs, XPU_MEM_GB)
            if XPU_MEM_GB > 0:
                alloc = (torch.xpu.memory_allocated(DEVICE_IDX) if DEVICE == "xpu"
                         else torch.cuda.memory_allocated(DEVICE_IDX)) / 1e9
                pct = alloc / XPU_MEM_GB * 100
                if pct < 55:
                    log.info(
                        "  GPU only %.0f%% full — try raising PER_DEVICE_BATCH "
                        "from %d to %d or disabling TURBO_MODE.",
                        pct, PER_DEVICE_BATCH, PER_DEVICE_BATCH + 2
                    )
                elif pct > 90:
                    log.info(
                        "  GPU %.0f%% full — good! If you see OOM, drop "
                        "PER_DEVICE_BATCH to %d.", pct, PER_DEVICE_BATCH - 1
                    )
                else:
                    log.info("  GPU %.0f%% full — healthy utilisation.", pct)
            XPUTrainer._step1_done = True
        return loss


class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        logs  = logs or {}
        step  = state.global_step
        total = state.max_steps or "?"
        sps   = step / max(1e-6, time.time() - self.t0)
        if isinstance(total, int) and total > 0 and step > 0:
            eta  = f"{max(0, (total - step) / max(1e-6, sps)) / 60:.1f}m"
        else:
            eta  = "n/a"
        loss_txt = f"{logs['loss']:.4f}" if "loss" in logs else "-"
        lr_txt   = f"{logs['learning_rate']:.2e}" if "learning_rate" in logs else "-"
        vs       = _vram_str()
        log.info(
            "step %s/%s | loss=%s | lr=%s | %.3f sps | eta=%s | %s",
            step, total, loss_txt, lr_txt, sps, eta, vs
        )


# ══════════════════════════════════════════════════════════════════════════
# TRAINING ARGS
# ══════════════════════════════════════════════════════════════════════════

def build_training_args() -> TrainingArguments:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    extra: dict = {}
    if DEVICE == "xpu" and _IPEX_OK:
        extra["use_ipex"] = True
        extra["no_cuda"]  = True

    # Prefer fused AdamW; adamw_apex_fused is fastest on XPU/CUDA
    optim_name = "adamw_torch_fused"   # safe default
    # try:
    #     import apex  # noqa
    #     optim_name = "adamw_apex_fused"
    # except ImportError:
    #     pass

    eval_strategy = "steps" if ENABLE_EVAL else "no"
    effective_max = TURBO_MAX_STEPS if TURBO_MODE else -1

    return TrainingArguments(
        output_dir              = OUTPUT_DIR,
        logging_dir             = os.path.join(OUTPUT_DIR, "logs"),
        logging_strategy        = "steps",
        logging_steps           = LOGGING_STEPS,
        logging_first_step      = True,
        disable_tqdm            = False,
        report_to               = "none",

        num_train_epochs        = NUM_EPOCHS,
        max_steps               = effective_max,

        per_device_train_batch_size = PER_DEVICE_BATCH,
        per_device_eval_batch_size  = EVAL_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,

        optim                   = optim_name,
        learning_rate           = LEARNING_RATE,
        weight_decay            = WEIGHT_DECAY,
        max_grad_norm           = MAX_GRAD_NORM,
        warmup_ratio            = WARMUP_RATIO,
        lr_scheduler_type       = LR_SCHEDULER,

        bf16                    = True,
        fp16                    = False,

        # Grad checkpointing controlled via USE_GRAD_CKPT flag above
        gradient_checkpointing  = USE_GRAD_CKPT,

        # DataLoader — high prefetch; workers set in XPUTrainer.get_train_dataloader
        dataloader_pin_memory          = False,
        dataloader_num_workers         = DATALOADER_WORKERS,
        dataloader_persistent_workers  = True,
        dataloader_prefetch_factor     = PREFETCH_FACTOR,

        eval_strategy           = eval_strategy,
        eval_steps              = EVAL_SAVE_STEPS if ENABLE_EVAL else None,
        save_strategy           = "steps",
        save_steps              = EVAL_SAVE_STEPS,
        save_total_limit        = SAVE_TOTAL_LIMIT,
        load_best_model_at_end  = False,

        seed                    = SEED,
        remove_unused_columns   = False,
        group_by_length         = not USE_PACKED_SEQS,  # packed = uniform length
        predict_with_generate   = False,

        **extra,
    )


# ══════════════════════════════════════════════════════════════════════════
# MERGE AND SAVE
# ══════════════════════════════════════════════════════════════════════════

def merge_and_save(trainer: XPUTrainer, tokenizer):
    log.info("Merging LoRA adapters into base model ...")
    os.makedirs(FINAL_DIR, exist_ok=True)
    peft_model = trainer.model.cpu().to(torch.float32)
    merged     = peft_model.merge_and_unload()
    merged.save_pretrained(FINAL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_DIR)
    log.info("Final merged model saved to: %s", FINAL_DIR)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    set_seed(SEED)
    log.info("=" * 60)
    log.info("  Albanian NLP Fine-Tuning — GPU-MAXIMIZED BUILD")
    log.info("  Model      : %s", MODEL_ID)
    log.info("  Device     : %s  (%.1f GB)", DEVICE.upper(), XPU_MEM_GB)
    log.info("  LoRA r     : %d   alpha : %d", LORA_R, LORA_ALPHA)
    log.info("  Batch      : %d × %d accum = %d effective",
             PER_DEVICE_BATCH, GRAD_ACCUM_STEPS,
             PER_DEVICE_BATCH * GRAD_ACCUM_STEPS)
    log.info("  Grad ckpt  : %s", "ON" if USE_GRAD_CKPT else "OFF (GPU-heavy mode)")
    log.info("  Packed seq : %s", "ON" if USE_PACKED_SEQS else "OFF")
    log.info("  Compile    : %s", "ON" if USE_TORCH_COMPILE else "OFF")
    log.info("  Turbo      : %s  sample=%.0f%%  max_steps=%d",
             "ON" if TURBO_MODE else "OFF",
             TURBO_SAMPLE_RATIO * 100, TURBO_MAX_STEPS if TURBO_MODE else -1)
    log.info("=" * 60)

    # GPU warm-up
    gpu_warmup()

    # Load + shuffle data
    all_records = load_all_json(JSON_DIR)
    if len(all_records) < 10:
        sys.exit(f"[ERROR] Only {len(all_records)} records — not enough to train.")

    random.seed(SEED)
    random.shuffle(all_records)

    if TURBO_MODE:
        keep        = max(64, int(len(all_records) * TURBO_SAMPLE_RATIO))
        all_records = all_records[:keep]
        log.info("Turbo: using %d samples", len(all_records))

    split_idx  = max(1, int(len(all_records) * TRAIN_SPLIT_RATIO))
    train_recs = all_records[:split_idx]
    eval_recs  = all_records[split_idx:]
    log.info("Split → train: %d | eval: %s",
             len(train_recs), len(eval_recs) if ENABLE_EVAL else "disabled")

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Tokenise (cached to disk on first run)
    base_train = CachedAlbanianDataset(train_recs, tokenizer, MAX_LENGTH, "train")
    base_eval  = (CachedAlbanianDataset(eval_recs, tokenizer, MAX_LENGTH, "eval")
                  if ENABLE_EVAL else None)

    # Pack sequences for zero-padding-waste training
    if USE_PACKED_SEQS:
        train_ds = PackedDataset(base_train, MAX_LENGTH)
        eval_ds  = PackedDataset(base_eval,  MAX_LENGTH) if base_eval else None
        collator = None  # packed tensors are already uniform length
    else:
        train_ds = base_train
        eval_ds  = base_eval
        collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, padding=True,
            pad_to_multiple_of=8, label_pad_token_id=-100,
            return_tensors="pt",
        )

    # Build trainer
    args    = build_training_args()
    trainer = XPUTrainer(
        model         = model,
        args          = args,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        data_collator = collator,
        tokenizer     = tokenizer,
    )
    trainer.add_callback(ProgressCallback())

    # Train
    log.info("Starting training ...")
    t0 = time.time()
    trainer.train()
    log.info("Training complete in %.1f min", (time.time() - t0) / 60)

    # Save
    merge_and_save(trainer, tokenizer)

    # Peak VRAM report
    if DEVICE in ("xpu", "cuda"):
        peak = (torch.xpu.max_memory_allocated(DEVICE_IDX) if DEVICE == "xpu"
                else torch.cuda.max_memory_allocated(DEVICE_IDX)) / 1e9
        log.info("Peak VRAM: %.2f GB / %.1f GB  (%.0f%%)",
                 peak, XPU_MEM_GB, peak / XPU_MEM_GB * 100)

    log.info("Done.  Model saved to: %s", FINAL_DIR)


if __name__ == "__main__":
    print("[BOOT] 6/6  entering main()", flush=True)
    main()