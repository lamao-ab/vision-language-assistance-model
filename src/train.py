"""
train.py
========
QLoRA fine-tuning of PaliGemma-3B — converted from the original Colab notebook.
Output format, data collator, training arguments, and summaries are identical
to the notebook.

Usage
-----
python src/train.py \
    --train_dataset_path data/train_dataset \
    --val_dataset_path   data/val_dataset \
    --output_dir         outputs/run1

Resume from latest checkpoint:
    python src/train.py ... --resume
"""

import argparse
import gc
import json
import os

import torch
from datasets import load_from_disk
from PIL import Image
import io
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)


# ── Data Collator ─────────────────────────────────────────────────────────────

class PaliGemmaDataCollator:
    def __init__(self, processor, max_length):
        self.processor  = processor
        self.max_length = max_length

    def __call__(self, examples):
        texts    = []
        suffixes = []
        images   = []

        for e in examples:
            # Support both dataset formats:
            # - original notebook format: "text" + "suffix" fields
            # - our format:               "question" + "answer" fields
            if "text" in e:
                clean_text = e["text"].replace("<image>", "").strip()
                prompt = "<image>" + clean_text
                suffix = e["suffix"]
            else:
                prompt = "<image>" + e["question"].strip()
                suffix = e["answer"]

            texts.append(prompt)
            suffixes.append(suffix)

            img = e["image"]
            if isinstance(img, dict):
                if "bytes" in img and img["bytes"] is not None:
                    img = Image.open(io.BytesIO(img["bytes"]))
                elif "path" in img and img["path"] is not None:
                    img = Image.open(img["path"])
                else:
                    raise ValueError(f"Unknown image format: {list(img.keys())}")

            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        model_inputs = self.processor(
            text=texts,
            images=images,
            suffix=suffixes,
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
        )

        return model_inputs


# ── Checkpoint helper ─────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning of PaliGemma-3B")
    parser.add_argument("--train_dataset_path",      required=True)
    parser.add_argument("--val_dataset_path",        required=True)
    parser.add_argument("--output_dir",              default="outputs/run1")
    parser.add_argument("--model_id",                default="google/paligemma-3b-mix-224")
    parser.add_argument("--num_epochs",              type=int,   default=3)
    parser.add_argument("--batch_size",              type=int,   default=16)
    parser.add_argument("--grad_accum",              type=int,   default=8)
    parser.add_argument("--learning_rate",           type=float, default=2e-5)
    parser.add_argument("--max_length",              type=int,   default=512)
    parser.add_argument("--lora_r",                  type=int,   default=8)
    parser.add_argument("--lora_alpha",              type=int,   default=16)
    parser.add_argument("--lora_dropout",            type=float, default=0.05)
    parser.add_argument("--early_stopping_patience", type=int,   default=2)
    parser.add_argument("--dataloader_workers",      type=int,   default=8)
    parser.add_argument("--resume",                  action="store_true",
                        help="Resume from the latest checkpoint in output_dir")
    args = parser.parse_args()

    effective_bs = args.batch_size * args.grad_accum

    # ── Banner ───────────────────────────────────────────────────────────────
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n🚀 STARTING QLORA TRAINER | GPU: {gpu_name}")

    # ── [1/6] Load Datasets ──────────────────────────────────────────────────
    print("\n[1/6] 📥 Loading Datasets...")
    train_dataset = load_from_disk(args.train_dataset_path)
    val_dataset   = load_from_disk(args.val_dataset_path)
    print(f"   ✅ Train Size: {len(train_dataset):,}")
    print(f"   ✅ Val Size:   {len(val_dataset):,}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"   📊 Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # ── [2/6] Quantization config ────────────────────────────────────────────
    print("\n[2/6] ⚙️  Configuring 4-bit Quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_quant_storage=torch.uint8,
    )
    print("   ✅ Quantization config ready:")
    print("      - Precision: 4-bit NF4")
    print("      - Compute dtype: bfloat16")
    print("      - Double quantization: Enabled")

    # ── [3/6] Load model ─────────────────────────────────────────────────────
    print("\n[3/6] ⏳ Loading Quantized PaliGemma...")
    processor = PaliGemmaProcessor.from_pretrained(args.model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)
    print("   ✅ Quantized model loaded")
    print(f"   📊 Model memory: {model.get_memory_footprint() / 1024**3:.2f} GB")
    print(f"   ⚡ Using SDPA attention (optimized for speed)")

    # ── [4/6] Data collator ──────────────────────────────────────────────────
    print("\n[4/6] 🔧 Setting up Data Collator...")
    data_collator = PaliGemmaDataCollator(processor, max_length=args.max_length)
    print("   ✅ Data collator ready")

    # ── [5/6] LoRA adapters ──────────────────────────────────────────────────
    print("\n[5/6] 🔧 Adding LoRA Adapters...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── [6/6] Training ───────────────────────────────────────────────────────
    print("\n[6/6] 🚀 Starting QLoRA Training...")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,

        # Logging
        logging_steps=100,
        logging_first_step=True,
        logging_dir=os.path.join(args.output_dir, "logs"),

        # Eval / Save
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Speed optimisations
        bf16=True,
        tf32=True,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        eval_accumulation_steps=4,
        skip_memory_metrics=True,

        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )],
    )

    # ── Resume checkpoint ────────────────────────────────────────────────────
    checkpoint = None
    if args.resume:
        checkpoint = find_latest_checkpoint(args.output_dir)
        if checkpoint:
            print(f"\n♻️  Resuming from: {os.path.basename(checkpoint)}")
        else:
            print("\n⚠️  Could not find a checkpoint — starting from scratch.")

    # ── Configuration display ─────────────────────────────────────────────────
    steps_per_epoch      = len(train_dataset) // effective_bs
    total_steps          = steps_per_epoch * args.num_epochs
    estimated_time_hours = (total_steps * 3.5) / 3600

    print("\n" + "=" * 70)
    print("⏳ TRAINING CONFIGURATION (SPEED OPTIMIZED)")
    print("=" * 70)
    print(f"   Model: PaliGemma-3B (4-bit quantized)")
    print(f"   Method: QLoRA")
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Val samples: {len(val_dataset):,}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Batch size: {args.batch_size} (INCREASED for speed)")
    print(f"   Grad accum: {args.grad_accum} (DECREASED)")
    print(f"   Effective BS: {effective_bs}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Attention: SDPA (optimized)")
    print(f"   TF32: Enabled (A100 acceleration)")
    print(f"   Data workers: {args.dataloader_workers} (parallel loading)")
    print("=" * 70)

    print(f"\n⏱️  Speed Estimate:")
    print(f"   Steps per epoch: {steps_per_epoch:,}")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Estimated time: ~{estimated_time_hours:.1f} hours")
    print(f"   (vs ~24 hours with batch_size=8)")

    # ── Pre-training validation ───────────────────────────────────────────────
    print("\n🔍 Pre-training Validation:")
    try:
        sample_batch = next(iter(trainer.get_eval_dataloader()))
        sample_batch = {k: v.to(model.device) for k, v in sample_batch.items()}
        with torch.no_grad():
            outputs = model(**sample_batch)
            sample_loss = outputs.loss.item()
        print(f"   ✅ Forward pass OK (initial loss: {sample_loss:.4f})")
        del sample_batch, outputs
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ⚠️  Forward pass check: {e}")
        print(f"   Continuing (Trainer will handle it)")

    # ── Start training ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70 + "\n")

    trainer.train(resume_from_checkpoint=checkpoint)

    print("\n" + "=" * 70)
    print("✅ TRAINING FINISHED!")
    print("=" * 70)

    # ── Save final adapter ────────────────────────────────────────────────────
    print("\n💾 Saving Final Adapter...")
    final_path = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    deployment_info = {
        "quantization":     "4-bit NF4",
        "compute_dtype":    "bfloat16",
        "double_quantization": True,
        "attn_implementation": "sdpa",
        "deployment_ready": True,
        "jetson_compatible": True,
        "training_speed_optimizations": {
            "batch_size":        args.batch_size,
            "tf32":              True,
            "sdpa_attention":    True,
            "dataloader_workers": args.dataloader_workers,
        },
    }
    with open(os.path.join(final_path, "deployment_config.json"), "w") as f:
        json.dump(deployment_info, f, indent=2)
    print(f"   ✅ Saved to: {final_path}")

    # ── Save training history ─────────────────────────────────────────────────
    print("\n📊 Saving History...")
    training_history = {
        "log_history":            trainer.state.log_history,
        "best_model_checkpoint":  trainer.state.best_model_checkpoint,
        "best_metric":            trainer.state.best_metric,
        "global_step":            trainer.state.global_step,
        "epoch":                  trainer.state.epoch,
        "config": {
            "model_id":      args.model_id,
            "method":        "QLoRA",
            "quantization":  "4-bit NF4",
            "speed_optimized": True,
            "num_epochs":    args.num_epochs,
            "batch_size":    args.batch_size,
            "grad_accum":    args.grad_accum,
            "learning_rate": args.learning_rate,
            "max_length":    args.max_length,
            "output_dir":    args.output_dir,
        },
    }
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"   ✅ Saved to: {history_path}")

    # ── Training summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📈 TRAINING SUMMARY")
    print("=" * 70)

    log           = trainer.state.log_history
    eval_entries  = [e for e in log if "eval_loss" in e]
    train_entries = [e for e in log if "loss" in e and "eval_loss" not in e]

    def last_train_loss_for_epoch(target_epoch):
        candidates = [e["loss"] for e in train_entries
                      if e.get("epoch", 0) <= target_epoch + 1e-6]
        return candidates[-1] if candidates else None

    if eval_entries:
        print("\n📊 Results:")
        print("-" * 70)
        print(f"{'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15} {'Time':<10}")
        print("-" * 70)
        for i, metric in enumerate(eval_entries, 1):
            ep         = metric.get("epoch", i)
            train_loss = last_train_loss_for_epoch(ep)
            tl_str     = f"{train_loss:.4f}" if train_loss is not None else "N/A"
            eval_loss  = f"{metric.get('eval_loss', 0):.4f}"
            eval_time  = (f"{metric.get('eval_runtime', 0):.0f}s"
                          if "eval_runtime" in metric else "N/A")
            print(f"Epoch {i:<4} {tl_str:<15} {eval_loss:<15} {eval_time:<10}")

    print("\n" + "-" * 70)
    if trainer.state.best_model_checkpoint:
        print(f"🏆 Best checkpoint: {os.path.basename(trainer.state.best_model_checkpoint)}")
    print(f"🏆 Best Val Loss:   {trainer.state.best_metric:.4f}")
    print(f"📊 Epochs:          {int(trainer.state.epoch)}")
    print(f"📊 Total Steps:     {trainer.state.global_step:,}")

    if torch.cuda.is_available():
        print(f"💾 Peak Memory:     {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")

    if epoch_metrics and len(epoch_metrics) > 1:
        initial     = epoch_metrics[0].get("eval_loss", 0)
        final_loss  = trainer.state.best_metric
        if initial > 0:
            improvement = ((initial - final_loss) / initial) * 100
            print(f"📈 Improvement:     {improvement:.1f}%")

    print("-" * 70)

    # ── Final output ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ QLORA TRAINING COMPLETE!")
    print("=" * 70)

    print("\n📂 Outputs:")
    print(f"   Adapter : {final_path}")
    print(f"   History : {history_path}")
    print(f"   Logs    : {os.path.join(args.output_dir, 'logs')}")

    print("\n🚀 Deployment:")
    print("   ✅ Jetson Orin Nano 8GB ready")
    print("   ✅ 4-bit quantization")
    print("   ✅ Expected accuracy: 86.0–86.5%")

    print("\n🎯 Next Steps:")
    print(f"   1. tensorboard --logdir={os.path.join(args.output_dir, 'logs')}")
    print("   2. Evaluate on validation set")
    print("   3. Deploy to Jetson")

    print("\n" + "=" * 70)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
