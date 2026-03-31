"""
train_lora.py
=============
Standard LoRA fine-tuning of PaliGemma-3B (bfloat16, no quantisation).
Run once per rank value to produce one adapter per experiment.

Usage
-----
python src/train_lora.py \
    --train_dataset_path data/train_dataset \
    --val_dataset_path   data/val_dataset \
    --base_output_dir    outputs/lora \
    --lora_rank          8 \
    --num_epochs         3 \
    --batch_size         16 \
    --grad_accum         8

Supported ranks: 4, 8, 16, 32, 64
Each run saves to:  <base_output_dir>_rank_<lora_rank>/
"""

import argparse
import gc
import json
import os

import torch
from datasets import load_from_disk
from PIL import Image
import io
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Terminal formatting helpers  (matches train.py / prepare_dataset.py)
# ══════════════════════════════════════════════════════════════════════════════

W = 72

def _line(char="─"): return char * W
def _double():       return "═" * W

def section(title: str) -> None:
    print(f"\n{_double()}")
    print(f"  {title}")
    print(_double())

def step(n: int, total: int, title: str) -> None:
    print(f"\n{_line()}")
    print(f"  Step {n}/{total}  │  {title}")
    print(_line())

def info(label: str, value: str = "") -> None:
    if value:
        print(f"    {label:<30}  {value}")
    else:
        print(f"    {label}")

def ok(msg: str)   -> None: print(f"    [+]  {msg}")
def warn(msg: str) -> None: print(f"    [!]  {msg}")
def item(msg: str) -> None: print(f"    [-]  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
#  Data Collator
# ══════════════════════════════════════════════════════════════════════════════

class PaliGemmaDataCollator:
    def __init__(self, processor, max_length: int):
        self.processor  = processor
        self.max_length = max_length

    def __call__(self, examples):
        texts    = []
        suffixes = []
        images   = []

        for e in examples:
            clean_text = e["text"].replace("<image>", "").strip()
            prompt = "<image>" + clean_text + "\n"      # ← LoRA adds \n

            texts.append(prompt)
            suffixes.append(e["suffix"])

            img = e["image"]
            if isinstance(img, dict):
                if img.get("bytes") is not None:
                    img = Image.open(io.BytesIO(img["bytes"]))
                elif img.get("path") is not None:
                    img = Image.open(img["path"])
                else:
                    raise ValueError(f"Unknown image format: {list(img.keys())}")

            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        return self.processor(
            text=texts,
            images=images,
            suffix=suffixes,
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Checkpoint helper
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None
    try:
        checkpoints = [
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint") and d.split("-")[-1].isdigit()
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            return os.path.join(output_dir, latest)
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Standard LoRA fine-tuning of PaliGemma-3B")
    parser.add_argument("--train_dataset_path",   required=True)
    parser.add_argument("--val_dataset_path",     required=True)
    parser.add_argument("--base_output_dir",      default="outputs/lora")
    parser.add_argument("--model_id",             default="google/paligemma-3b-mix-224")
    parser.add_argument("--lora_rank",            type=int,   default=8,
                        choices=[4, 8, 16, 32, 64])
    parser.add_argument("--num_epochs",           type=int,   default=3)
    parser.add_argument("--batch_size",           type=int,   default=16)
    parser.add_argument("--grad_accum",           type=int,   default=8)
    parser.add_argument("--learning_rate",        type=float, default=2e-4)
    parser.add_argument("--max_length",           type=int,   default=512)
    parser.add_argument("--lora_dropout",         type=float, default=0.05)
    parser.add_argument("--dataloader_workers",   type=int,   default=4)
    parser.add_argument("--resume",               action="store_true",
                        help="Resume from the latest checkpoint in output_dir")
    args = parser.parse_args()

    # One output folder per rank — matches Colab structure
    output_dir   = f"{args.base_output_dir}_rank_{args.lora_rank}"
    lora_alpha   = args.lora_rank * 2          # auto-scales with rank
    effective_bs = args.batch_size * args.grad_accum

    # ── Banner ────────────────────────────────────────────────────────────────
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print(f"\n{_double()}")
    print(f"  PaliGemma-3B  ·  LoRA Fine-Tuning")
    print(f"  GPU  : {gpu_name}")
    print(f"  Rank : {args.lora_rank}  →  output: {output_dir}")
    print(_double())

    # ── Step 1 — Load Datasets ────────────────────────────────────────────────
    step(1, 5, "Loading Datasets")

    train_dataset = load_from_disk(args.train_dataset_path)
    val_dataset   = load_from_disk(args.val_dataset_path)

    ok(f"Train  {len(train_dataset):>10,} samples")
    ok(f"Val    {len(val_dataset):>10,} samples")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        ok(f"GPU memory (initial)   {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # ── Step 2 — Load Model ───────────────────────────────────────────────────
    step(2, 5, "Loading PaliGemma  (bfloat16)")

    processor = PaliGemmaProcessor.from_pretrained(args.model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        revision="bfloat16",
    )
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    model.to("cuda")

    ok(f"Model loaded  ({model.get_memory_footprint() / 1024**3:.2f} GB in VRAM)")
    info("Precision", "bfloat16  (no quantisation)")

    # ── Step 3 — Data Collator ────────────────────────────────────────────────
    step(3, 5, "Setting Up Data Collator")

    data_collator = PaliGemmaDataCollator(processor, max_length=args.max_length)
    ok("Data collator ready")

    # ── Step 4 — LoRA Adapters ────────────────────────────────────────────────
    step(4, 5, f"Attaching LoRA Adapters  (rank={args.lora_rank})")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,              # auto-scaled: rank × 2
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora_config)

    info("Rank (r)",       str(args.lora_rank))
    info("Alpha",          f"{lora_alpha}  (rank × 2)")
    info("Dropout",        str(args.lora_dropout))
    info("Target modules", "q/k/v/o_proj  ·  gate/up/down_proj")
    print()
    model.print_trainable_parameters()

    # ── Step 5 — Training Setup ───────────────────────────────────────────────
    step(5, 5, "Preparing Trainer")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,

        # Cosine scheduler + warmup
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        # Logging
        logging_steps=50,
        logging_first_step=True,
        logging_dir=os.path.join(output_dir, "logs"),

        # Eval / Save
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Precision
        bf16=True,

        # Memory optimisations (full bfloat16 needs these on a 22 GB GPU)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",

        # Data loading
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,

        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # ── Resume checkpoint ─────────────────────────────────────────────────────
    checkpoint = None
    if args.resume:
        checkpoint = find_latest_checkpoint(output_dir)
        if checkpoint:
            ok(f"Resuming from  {os.path.basename(checkpoint)}")
        else:
            warn("No checkpoint found — starting from scratch")

    # ── Training configuration table ──────────────────────────────────────────
    steps_per_epoch      = len(train_dataset) // effective_bs
    total_steps          = steps_per_epoch * args.num_epochs

    section("Training Configuration")

    C = 30
    print(f"    {'Model':<{C}}  PaliGemma-3B  (bfloat16)")
    print(f"    {'Method':<{C}}  LoRA")
    print(f"    {'Train samples':<{C}}  {len(train_dataset):,}")
    print(f"    {'Val samples':<{C}}  {len(val_dataset):,}")
    print()
    print(f"    {'Rank (r)':<{C}}  {args.lora_rank}")
    print(f"    {'Alpha':<{C}}  {lora_alpha}  (rank × 2)")
    print()
    print(f"    {'Epochs':<{C}}  {args.num_epochs}")
    print(f"    {'Batch size':<{C}}  {args.batch_size}")
    print(f"    {'Gradient accumulation':<{C}}  {args.grad_accum}")
    print(f"    {'Effective batch size':<{C}}  {effective_bs}")
    print(f"    {'Learning rate':<{C}}  {args.learning_rate:.2e}")
    print(f"    {'LR scheduler':<{C}}  cosine  (warmup=0.03)")
    print(f"    {'Max sequence length':<{C}}  {args.max_length}")
    print(f"    {'Dataloader workers':<{C}}  {args.dataloader_workers}")
    print(f"    {'Gradient checkpointing':<{C}}  enabled")
    print(f"    {'Optimizer':<{C}}  adamw_torch_fused")
    print(f"    {'Gradient checkpointing':<{C}}  enabled")
    print()
    print(f"    {'Steps per epoch':<{C}}  {steps_per_epoch:,}")
    print(f"    {'Total steps':<{C}}  {total_steps:,}")
    print(f"    {'Output dir':<{C}}  {output_dir}")

    # ── Start training ────────────────────────────────────────────────────────
    section("Training")

    trainer.train(resume_from_checkpoint=checkpoint)

    print(f"\n{_double()}")
    print(f"  Training Complete")
    print(_double())

    # ── Save final adapter ────────────────────────────────────────────────────
    print(f"\n{_line()}")
    print(f"  Saving Artefacts")
    print(_line())

    final_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    ok(f"Adapter saved    →  {final_path}")

    # ── Save training history ─────────────────────────────────────────────────
    training_history = {
        "log_history":           trainer.state.log_history,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric":           trainer.state.best_metric,
        "global_step":           trainer.state.global_step,
        "epoch":                 trainer.state.epoch,
        "config": {
            "model_id":      args.model_id,
            "method":        "LoRA (standard)",
            "precision":     "bfloat16",
            "rank":          args.lora_rank,
            "alpha":         lora_alpha,
            "num_epochs":    args.num_epochs,
            "batch_size":    args.batch_size,
            "grad_accum":    args.grad_accum,
            "learning_rate": args.learning_rate,
            "max_length":    args.max_length,
            "output_dir":    output_dir,
        },
    }
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    ok(f"History saved    →  {history_path}")

    # ── Training summary ──────────────────────────────────────────────────────
    section("Training Summary")

    log           = trainer.state.log_history
    eval_entries  = [e for e in log if "eval_loss" in e]
    train_entries = [e for e in log if "loss" in e and "eval_loss" not in e]

    def last_train_loss_for_epoch(target_epoch):
        candidates = [e["loss"] for e in train_entries
                      if e.get("epoch", 0) <= target_epoch + 1e-6]
        return candidates[-1] if candidates else None

    if eval_entries:
        print(f"\n    {'Epoch':<8}  {'Train Loss':>12}  {'Val Loss':>12}  {'Eval Time':>10}")
        print(f"    {'─'*8}  {'─'*12}  {'─'*12}  {'─'*10}")
        for i, metric in enumerate(eval_entries, 1):
            ep         = metric.get("epoch", i)
            train_loss = last_train_loss_for_epoch(ep)
            tl_str     = f"{train_loss:.4f}" if train_loss is not None else "—"
            eval_loss  = f"{metric.get('eval_loss', 0):.4f}"
            eval_time  = (f"{metric.get('eval_runtime', 0):.0f} s"
                          if "eval_runtime" in metric else "—")
            print(f"    {i:<8}  {tl_str:>12}  {eval_loss:>12}  {eval_time:>10}")

    print(f"\n    {_line()[:-4]}")

    if trainer.state.best_model_checkpoint:
        info("Best checkpoint",  os.path.basename(trainer.state.best_model_checkpoint))
    info("Best val loss",    f"{trainer.state.best_metric:.4f}")
    info("Epochs completed", str(int(trainer.state.epoch)))
    info("Total steps",      f"{trainer.state.global_step:,}")

    if torch.cuda.is_available():
        info("Peak GPU memory",  f"{torch.cuda.max_memory_allocated()/1024**3:.1f} GB")

    if eval_entries and len(eval_entries) > 1:
        initial    = eval_entries[0].get("eval_loss", 0)
        final_loss = trainer.state.best_metric
        if initial > 0:
            improvement = ((initial - final_loss) / initial) * 100
            info("Loss improvement", f"{improvement:.1f}%")

    # ── Output files ──────────────────────────────────────────────────────────
    print(f"\n{_line()}")
    print(f"  Output Files")
    print(_line())
    info("Adapter",  final_path)
    info("History",  history_path)
    info("Logs",     os.path.join(output_dir, "logs"))

    # ── Next steps ────────────────────────────────────────────────────────────
    print(f"\n{_line()}")
    print(f"  Next Steps")
    print(_line())
    item(f"tensorboard --logdir={os.path.join(output_dir, 'logs')}")
    item("Evaluate on validation set")
    item("Compare ranks:  rank_4 / rank_8 / rank_16 / rank_32 / rank_64")

    print(f"\n{_double()}\n")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
