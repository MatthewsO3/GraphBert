import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from model import GraphCodeBERTDataset, GraphCodeBERTWithEdgePrediction, MLMWithEdgePredictionCollator


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Performance tracker – now also tracks per-language eval losses
# ---------------------------------------------------------------------------

EVAL_LANGUAGES = ["python", "javascript", "java"]


class PerformanceTracker:
    def __init__(self, output_dir: str, patience: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.patience_counter = 0
        self.best_val_loss = float("inf")

        self.history: Dict = {
            "epoch": [],
            "train_total_loss": [],
            "train_mlm_loss": [],
            "train_edge_loss": [],
            "train_batch_losses": [],
            "train_mlm_batch_losses": [],
            "train_edge_batch_losses": [],
            "val_total_loss": [],
            "val_mlm_loss": [],
            "val_edge_loss": [],
            "val_batch_losses": [],
            "val_mlm_batch_losses": [],
            "val_edge_batch_losses": [],
            "learning_rate": [],
            "best_val_loss": None,
            "best_epoch": None,
        }
        # Per-language eval history
        for lang in EVAL_LANGUAGES:
            self.history[f"{lang}_total_loss"] = []
            self.history[f"{lang}_mlm_loss"] = []
            self.history[f"{lang}_edge_loss"] = []

    # ------------------------------------------------------------------
    def log_batch(self, phase: str, total_loss, mlm_loss, edge_loss):
        pfx = "train" if phase == "train" else "val"
        self.history[f"{pfx}_batch_losses"].append(total_loss)
        self.history[f"{pfx}_mlm_batch_losses"].append(mlm_loss if mlm_loss else 0)
        self.history[f"{pfx}_edge_batch_losses"].append(edge_loss if edge_loss else 0)

    def log_epoch(self, epoch: int, phase: str, total_loss, mlm_loss, edge_loss, lr=None):
        if phase == "train":
            self.history["epoch"].append(epoch)
            self.history["train_total_loss"].append(total_loss)
            self.history["train_mlm_loss"].append(mlm_loss)
            self.history["train_edge_loss"].append(edge_loss)
            if lr is not None:
                self.history["learning_rate"].append(lr)
        else:
            self.history["val_total_loss"].append(total_loss)
            self.history["val_mlm_loss"].append(mlm_loss)
            self.history["val_edge_loss"].append(edge_loss)

    def log_language_eval(self, lang: str, total_loss: float, mlm_loss: float, edge_loss: float):
        """Store per-language evaluation results for one epoch."""
        self.history[f"{lang}_total_loss"].append(total_loss)
        self.history[f"{lang}_mlm_loss"].append(mlm_loss)
        self.history[f"{lang}_edge_loss"].append(edge_loss)

    # ------------------------------------------------------------------
    def update_best(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.history["best_val_loss"] = val_loss
            self.history["best_epoch"] = epoch
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        return False

    def should_stop_early(self) -> bool:
        return self.patience_counter >= self.patience

    # ------------------------------------------------------------------
    def _compute_summary(self) -> Dict:
        summary: Dict = {
            "total_epochs": len(self.history["epoch"]),
            "best_epoch": self.history["best_epoch"],
            "best_val_loss": self.history["best_val_loss"],
            "final_train_loss": self.history["train_total_loss"][-1] if self.history["train_total_loss"] else None,
            "final_val_loss": self.history["val_total_loss"][-1] if self.history["val_total_loss"] else None,
            "min_train_loss": min(self.history["train_total_loss"]) if self.history["train_total_loss"] else None,
            "min_val_loss": min(self.history["val_total_loss"]) if self.history["val_total_loss"] else None,
            "final_train_mlm_loss": self.history["train_mlm_loss"][-1] if self.history["train_mlm_loss"] else None,
            "final_train_edge_loss": self.history["train_edge_loss"][-1] if self.history["train_edge_loss"] else None,
            "final_val_mlm_loss": self.history["val_mlm_loss"][-1] if self.history["val_mlm_loss"] else None,
            "final_val_edge_loss": self.history["val_edge_loss"][-1] if self.history["val_edge_loss"] else None,
            "total_batches_train": len(self.history["train_batch_losses"]),
            "total_batches_val": len(self.history["val_batch_losses"]),
        }
        for lang in EVAL_LANGUAGES:
            losses = self.history.get(f"{lang}_total_loss", [])
            summary[f"{lang}_final_loss"] = losses[-1] if losses else None
            summary[f"{lang}_min_loss"] = min(losses) if losses else None
        return summary

    def _save_history_json(self):
        path = self.output_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {path}")

    def _save_summary_json(self):
        summary = self._compute_summary()
        path = self.output_dir / "training_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved training summary to {path}")

    def _save_metrics_csv(self):
        try:
            path = self.output_dir / "training_metrics.csv"
            lang_total_cols = [f"{l}_total_loss" for l in EVAL_LANGUAGES]
            lang_mlm_cols   = [f"{l}_mlm_loss"   for l in EVAL_LANGUAGES]
            lang_edge_cols  = [f"{l}_edge_loss"   for l in EVAL_LANGUAGES]

            header = (
                ["Epoch", "Train Total", "Train MLM", "Train Edge",
                 "Val Total", "Val MLM", "Val Edge", "Learning Rate"]
                + lang_total_cols + lang_mlm_cols + lang_edge_cols
            )
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(len(self.history["epoch"])):
                    row = [
                        self.history["epoch"][i],
                        self.history["train_total_loss"][i],
                        self.history["train_mlm_loss"][i],
                        self.history["train_edge_loss"][i],
                        self.history["val_total_loss"][i] if i < len(self.history["val_total_loss"]) else "",
                        self.history["val_mlm_loss"][i]   if i < len(self.history["val_mlm_loss"])   else "",
                        self.history["val_edge_loss"][i]  if i < len(self.history["val_edge_loss"])  else "",
                        self.history["learning_rate"][i]  if i < len(self.history["learning_rate"])  else "",
                    ]
                    for col in lang_total_cols + lang_mlm_cols + lang_edge_cols:
                        vals = self.history.get(col, [])
                        row.append(vals[i] if i < len(vals) else "")
                    writer.writerow(row)
            print(f"Saved metrics CSV to {path}")
        except Exception as e:
            print(f"Could not save CSV: {e}")

    def save(self):
        self._save_history_json()
        self._save_summary_json()
        self._save_metrics_csv()


# ---------------------------------------------------------------------------
# Checkpoint manager (unchanged)
# ---------------------------------------------------------------------------

class ModelCheckpointManager:
    def __init__(self, output_dir: str, keep_last_n: int = 999):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir = self.output_dir / "best_model"
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoint_list: List[Path] = []

    def save_checkpoint(self, model, tokenizer, epoch: int):
        ckpt_dir = self.checkpoints_dir / f"epoch_{epoch:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        self.checkpoint_list.append(ckpt_dir)
        if len(self.checkpoint_list) > self.keep_last_n:
            old = self.checkpoint_list.pop(0)
            import shutil
            shutil.rmtree(old)
            print(f"Removed old checkpoint: {old}")
        print(f"Saved checkpoint to {ckpt_dir}")

    def save_best_model(self, model, tokenizer):
        model.save_pretrained(str(self.best_model_dir))
        tokenizer.save_pretrained(str(self.best_model_dir))
        print(f"Saved best model to {self.best_model_dir}")

    def get_best_model_path(self) -> str:
        return str(self.best_model_dir)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
        use_amp = False
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
        use_amp = True
    else:
        device = torch.device("cpu")
        print("Using CPU")
        use_amp = False
    return device, use_amp


def clear_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Config / project root helpers
# ---------------------------------------------------------------------------

def find_project_root(start_path: Optional[Path] = None) -> Path:
    if start_path is None:
        start_path = Path(__file__).parent.absolute()
    current = start_path
    while True:
        if (current / "config.json").exists():
            return current
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                "Could not find project root (no config.json found)."
            )
        current = parent


def load_config_and_set_defaults(parser) -> Path:
    project_root = find_project_root()
    config_path = project_root / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        parser.set_defaults(**config_data.get("train", {}))
        print(f"Loaded config from: {config_path}")
    else:
        raise FileNotFoundError(f"config.json not found at {config_path}")
    return project_root


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_dataloader(
    data_path: Path,
    tokenizer,
    args,
    batch_size: int,
    shuffle: bool = False,
    num_workers: Optional[int] = None,
) -> DataLoader:
    """Build a DataLoader for *any* val.jsonl / val.json file."""
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
    dataset = GraphCodeBERTDataset(str(data_path), tokenizer, args.max_length)
    collator = MLMWithEdgePredictionCollator(tokenizer, mlm_probability=args.mlm_probability)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )


def setup_model_and_data(args, device, project_root: Path):
    print("Loading GraphCodeBERT…")

    checkpoint_path = getattr(args, "checkpoint_path", None)
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = GraphCodeBERTWithEdgePrediction.from_pretrained(checkpoint_path).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(checkpoint_path)
    else:
        print("Loading base model…")
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
        model = GraphCodeBERTWithEdgePrediction("microsoft/graphcodebert-base").to(device)

    print("Model loaded successfully")

    # Primary train/val data
    data_path = project_root / args.data_file
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Check data_file in config.json (relative to project root: {project_root})"
        )

    full_dataset = GraphCodeBERTDataset(str(data_path), tokenizer, args.max_length)
    val_size = int(args.validation_split * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [len(full_dataset) - val_size, val_size]
    )

    collator = MLMWithEdgePredictionCollator(tokenizer, mlm_probability=args.mlm_probability)
    workers = min(4, os.cpu_count() or 1)

    train_dl = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=workers, pin_memory=True,
    )
    val_dl = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        collate_fn=collator, num_workers=workers, pin_memory=True,
    )

    # Per-language eval dataloaders
    lang_dataloaders: Dict[str, Optional[DataLoader]] = {}
    for lang in EVAL_LANGUAGES:
        # Support both val.json and val.jsonl
        for fname in ("val.jsonl", "val.json"):
            lang_path = project_root / "data" / lang / fname
            if lang_path.exists():
                try:
                    lang_dataloaders[lang] = build_dataloader(
                        lang_path, tokenizer, args,
                        batch_size=args.batch_size * 2,
                        shuffle=False,
                    )
                    print(f"  [{lang}] eval data loaded from {lang_path}")
                except Exception as exc:
                    print(f"  [{lang}] WARNING – could not load {lang_path}: {exc}")
                    lang_dataloaders[lang] = None
                break
        else:
            print(f"  [{lang}] WARNING – no val file found under data/{lang}/; skipping.")
            lang_dataloaders[lang] = None

    return model, tokenizer, train_dl, val_dl, lang_dataloaders


# ---------------------------------------------------------------------------
# Optimiser / scheduler
# ---------------------------------------------------------------------------

def setup_optimizer_and_scheduler(model, args, train_dl):
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_dl) * args.epochs,
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Train / validate helpers
# ---------------------------------------------------------------------------

def _run_batch(model, batch, device, use_amp: bool):
    """Forward pass – returns (loss, mlm_loss, edge_loss) tensors."""
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    kwargs = dict(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        labels=batch["labels"],
        edge_batch_idx=batch["edge_batch_idx"],
        edge_node1_pos=batch["edge_node1_pos"],
        edge_node2_pos=batch["edge_node2_pos"],
        edge_labels=batch["edge_labels"],
    )

    if use_amp:
        dtype = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=dtype):
            outputs = model(**kwargs)
    else:
        outputs = model(**kwargs)

    return outputs["loss"], outputs["mlm_loss"], outputs["edge_loss"]


def train_epoch(
    model, dataloader, optimizer, scheduler, device,
    tracker: PerformanceTracker, scaler, use_amp: bool = False,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = total_mlm = total_edge = 0.0
    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()
        try:
            loss, mlm_loss, edge_loss = _run_batch(model, batch, device, use_amp)

            if use_amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            if mlm_loss:  total_mlm  += mlm_loss.item()
            if edge_loss: total_edge += edge_loss.item()
            batch_count += 1

            tracker.log_batch(
                "train", loss.item(),
                mlm_loss.item() if mlm_loss else None,
                edge_loss.item() if edge_loss else None,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg":  f"{total_loss / batch_count:.4f}",
                "mlm":  f"{mlm_loss.item() if mlm_loss else 0:.4f}",
                "edge": f"{edge_loss.item() if edge_loss else 0:.4f}",
                "lr":   f"{current_lr:.2e}",
            })

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(
                    f"\nOUT OF MEMORY – batch size: {batch['input_ids'].shape[0]}, "
                    f"seq len: {batch['input_ids'].shape[1]}"
                )
                raise
            raise
        finally:
            clear_cache(device)

    n = max(batch_count, 1)
    return total_loss / n, total_mlm / n, total_edge / n


def validate(
    model, dataloader, device,
    tracker: PerformanceTracker, use_amp: bool = False,
    phase: str = "val",
) -> Tuple[float, float, float]:
    """Generic evaluation loop – works for the primary val set and per-language sets."""
    model.eval()
    total_loss = total_mlm = total_edge = 0.0
    batch_count = 0
    progress_bar = tqdm(dataloader, desc=f"Eval [{phase}]")

    with torch.no_grad():
        for batch in progress_bar:
            loss, mlm_loss, edge_loss = _run_batch(model, batch, device, use_amp)

            total_loss += loss.item()
            if mlm_loss:  total_mlm  += mlm_loss.item()
            if edge_loss: total_edge += edge_loss.item()
            batch_count += 1

            if phase == "val":
                tracker.log_batch(
                    "val", loss.item(),
                    mlm_loss.item() if mlm_loss else None,
                    edge_loss.item() if edge_loss else None,
                )

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg":  f"{total_loss / batch_count:.4f}",
                "mlm":  f"{mlm_loss.item() if mlm_loss else 0:.4f}",
                "edge": f"{edge_loss.item() if edge_loss else 0:.4f}",
            })

    n = max(batch_count, 1)
    return total_loss / n, total_mlm / n, total_edge / n


def evaluate_all_languages(
    model, lang_dataloaders: Dict[str, Optional[DataLoader]],
    device, tracker: PerformanceTracker, use_amp: bool,
) -> Dict[str, Tuple[float, float, float]]:
    """Run eval on each language's validation set and log results."""
    results: Dict[str, Tuple[float, float, float]] = {}
    for lang, dl in lang_dataloaders.items():
        if dl is None:
            print(f"  [{lang}] skipped (no data)")
            continue
        t, m, e = validate(model, dl, device, tracker, use_amp, phase=lang)
        tracker.log_language_eval(lang, t, m, e)
        results[lang] = (t, m, e)
    return results


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_training_config(args, device, use_amp: bool):
    print("\n--- Training Configuration ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  use_amp: {use_amp}")
    print(f"  device:  {device}")
    print("------------------------------\n")


def print_epoch_results(
    epoch: int, args,
    train_loss, train_mlm, train_edge,
    val_loss,   val_mlm,   val_edge,
    lang_results: Dict[str, Tuple[float, float, float]],
    current_lr: float,
    tracker: PerformanceTracker,
):
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"Epoch {epoch + 1} Results:")
    print(f"  Train  – Total: {train_loss:.6f}  MLM: {train_mlm:.6f}  Edge: {train_edge:.6f}")
    print(f"  Val    – Total: {val_loss:.6f}  MLM: {val_mlm:.6f}  Edge: {val_edge:.6f}")
    if lang_results:
        print(f"  ── Cross-language evaluation ──────────────────────────────────")
        for lang, (t, m, e) in lang_results.items():
            print(f"  {lang:<12} Total: {t:.6f}  MLM: {m:.6f}  Edge: {e:.6f}")
    print(f"  Learning Rate: {current_lr:.6e}")
    best_epoch_display = (tracker.history["best_epoch"] + 1
                          if tracker.history["best_epoch"] is not None else "N/A")
    print(f"  Best Val Loss: {tracker.best_val_loss:.6f} (Epoch {best_epoch_display})")
    print(f"  Patience:      {tracker.patience_counter}/{args.early_stopping_patience}")
    print(sep)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def training_loop(
    model, train_dl, val_dl,
    lang_dataloaders: Dict[str, Optional[DataLoader]],
    optimizer, scheduler,
    device, args, tracker: PerformanceTracker,
    checkpoint_manager: ModelCheckpointManager,
    scaler, use_amp: bool,
):
    for epoch in range(args.epochs):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 70}")

        clear_cache(device)

        # ---- Training ----
        train_loss, train_mlm, train_edge = train_epoch(
            model, train_dl, optimizer, scheduler, device, tracker, scaler, use_amp
        )

        clear_cache(device)

        # ---- Primary validation (same distribution as training data) ----
        val_loss, val_mlm, val_edge = validate(
            model, val_dl, device, tracker, use_amp, phase="val"
        )

        clear_cache(device)

        # ---- Per-language evaluation ----
        lang_results = evaluate_all_languages(
            model, lang_dataloaders, device, tracker, use_amp
        )

        clear_cache(device)

        # ---- Bookkeeping ----
        current_lr = optimizer.param_groups[0]["lr"]
        tracker.log_epoch(epoch, "train", train_loss, train_mlm, train_edge, current_lr)
        tracker.log_epoch(epoch, "val",   val_loss,   val_mlm,   val_edge)

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"\n  Peak GPU Memory: {peak_mem:.2f} GB")

        print_epoch_results(
            epoch, args,
            train_loss, train_mlm, train_edge,
            val_loss,   val_mlm,   val_edge,
            lang_results,
            current_lr, tracker,
        )

        # ---- Checkpoint & early stopping ----
        checkpoint_manager.save_checkpoint(model, args.tokenizer, epoch)

        if tracker.update_best(val_loss, epoch):
            print("\nNew best model! Saving…")
            checkpoint_manager.save_best_model(model, args.tokenizer)
        else:
            print(f"\nNo improvement. Patience: {tracker.patience_counter}/{args.early_stopping_patience}")

        if tracker.should_stop_early():
            print(
                f"\nEarly stopping triggered – no improvement for "
                f"{args.early_stopping_patience} epochs.\n"
                f"Best loss: {tracker.best_val_loss:.6f} at epoch "
                f"{tracker.history['best_epoch'] + 1}"
            )
            break

    print(f"\n{'=' * 70}")
    print(f"Training complete!  Best val loss: {tracker.best_val_loss:.6f} "
          f"at epoch {tracker.history['best_epoch'] + 1}")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GraphCodeBERT with Edge Prediction")
    parser.add_argument("--data_file",               type=str,   default=None)
    parser.add_argument("--output_dir",              type=str,   default=None)
    parser.add_argument("--checkpoint_path",         type=str,   default=None)
    parser.add_argument("--batch_size",              type=int,   default=None)
    parser.add_argument("--epochs",                  type=int,   default=None)
    parser.add_argument("--learning_rate",           type=float, default=None)
    parser.add_argument("--max_length",              type=int,   default=None)
    parser.add_argument("--warmup_steps",            type=int,   default=None)
    parser.add_argument("--mlm_probability",         type=float, default=None)
    parser.add_argument("--validation_split",        type=float, default=None)
    parser.add_argument("--weight_decay",            type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int,   default=3)
    parser.add_argument("--use_amp",                 action="store_true")

    project_root = load_config_and_set_defaults(parser)
    args = parser.parse_args()

    if not args.data_file:
        parser.error("data_file must be specified in config.json or via --data_file.")
    if not args.output_dir:
        parser.error("output_dir must be specified in config.json or via --output_dir.")

    set_seed(42)

    device, use_amp = setup_device()
    output_path = project_root / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    tracker = PerformanceTracker(str(output_path), patience=args.early_stopping_patience)
    checkpoint_manager = ModelCheckpointManager(str(output_path))

    model, tokenizer, train_dl, val_dl, lang_dataloaders = setup_model_and_data(
        args, device, project_root
    )
    args.tokenizer = tokenizer

    optimizer, scheduler = setup_optimizer_and_scheduler(model, args, train_dl)
    scaler = GradScaler() if use_amp else None

    print_training_config(args, device, use_amp)

    training_loop(
        model, train_dl, val_dl, lang_dataloaders,
        optimizer, scheduler, device, args, tracker,
        checkpoint_manager, scaler, use_amp,
    )

    print("=" * 70)
    print("SAVING PERFORMANCE METRICS…")
    print("=" * 70)
    tracker.save()
    print(f"\nAll results saved to: {output_path}")
    print(f"Best model at:        {checkpoint_manager.get_best_model_path()}")
    print(f"Checkpoints at:       {checkpoint_manager.checkpoints_dir}")


if __name__ == "__main__":
    main()