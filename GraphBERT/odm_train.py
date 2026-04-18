"""
train_odm.py  —  GraphCodeBERT training with Online Data Mixing (ODM)
Based on: "Efficient Online Data Mixing For Language Model Pre-Training"
          Albalak et al., 2023  (arXiv:2312.02406)

ODM overview
────────────
Instead of sampling C++ and Erlang data at fixed ratios, we treat each
language as a "bandit arm" and dynamically reweight sampling probabilities
during training so that the language the model currently finds hardest
(highest loss = most information gain) gets sampled more often.

Algorithm (Exp3-variant with moving-average reward):
  1. Maintain a per-domain estimated reward R̂_i (moving average of
     importance-weighted loss).
  2. At each step t, compute exploration rate:
         E_t = min(1/K, sqrt(ln K / (K * t)))
  3. Mix Gibbs distribution with uniform:
         π_t(D_i) = (1 - K*E_t) * softmax(E_{t-1} * R̂)[i]  +  E_t
  4. Sample a domain according to π_t, compute the batch loss.
  5. Update R̂_i ← α * R̂_i  +  (1-α) * loss / π_{t-1}(D_i)
  6. Repeat — no extra forward/backward passes needed.

Data domains (K=2 by default, easily extensible):
  - C++    (codeparrot/github-code-clean style JSONL)
  - Erlang (your existing pipeline JSONL)

Usage
─────
  python train_odm.py                          # uses config.json
  python train_odm.py --cpp_file   data/cpp_functions.jsonl \
                      --erl_file   data/erlang_functions.jsonl \
                      --output_dir runs/odm_v1
  python train_odm.py --checkpoint_path models/pretrained   # continue from ckpt

Fixes applied vs original
──────────────────────────
  #1  Removed double autocast — forward_pass owns it; training loop must not re-wrap.
  #2  Added scaler.unscale_(optimizer) before clip_grad_norm_ under AMP.
  #3  odm_warmup/odm_alpha/log_interval use `or` fallback so None from argparse
      doesn't silently propagate and crash.
  #4  _exploration_rate guards K==1 (log(1)==0 → E permanently 0 → no exploration).
  #5  Removed unused `Subset` import.
  #6  pin_memory gated on device.type=="cuda"; safe on MPS/CPU.
  #7  `epoch` initialised to -1 before the loop so post-loop summary never
      raises NameError if the loop body never executes.
  +   All CLI args default to None so `or` fallback chains work correctly.
  +   Checkpoint now also saves optimizer/scheduler/scaler/bandit state in
      training_state.pt alongside the model weights for full resumability.
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader          # FIX #5: Subset removed (unused)
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# Re-use existing model & dataset classes unchanged
from model import (
    GraphCodeBERTDataset,
    GraphCodeBERTWithEdgePrediction,
    MLMWithEdgePredictionCollator,
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_project_root(start_path: Path = None) -> Path:
    if start_path is None:
        start_path = Path(__file__).parent.absolute()
    current = start_path
    while True:
        if (current / "config.json").exists():
            return current
        parent = current.parent
        if parent == current:
            raise FileNotFoundError("config.json not found in any parent directory.")
        current = parent


def load_config() -> Dict:
    root = find_project_root()
    with open(root / "config.json") as f:
        return json.load(f)


def setup_device():
    if torch.backends.mps.is_available():
        return torch.device("mps"), False   # MPS: no AMP
    if torch.cuda.is_available():
        return torch.device("cuda"), True   # CUDA: AMP supported
    return torch.device("cpu"), False


# ─────────────────────────────────────────────────────────────────────────────
# ODM Bandit
# ─────────────────────────────────────────────────────────────────────────────

class ODMBandit:
    """
    Implements the Exp3-variant multi-armed bandit from Algorithm 1 of the
    ODM paper with a moving-average estimated reward instead of cumulative.

    Parameters
    ----------
    domain_names : list[str]
        Human-readable names for logging (e.g. ["cpp", "erlang"]).
    alpha : float
        Moving-average decay for reward estimates (0.9 → recent losses
        matter more; 0.99 → slower adaptation).
    warmup_steps : int
        Number of steps during which π stays uniform (the model is too
        noisy early on for loss differences to be meaningful).
    initial_weights : list[float] | None
        Optional starting weights; normalised to sum to 1. Defaults to uniform.
    """

    def __init__(
        self,
        domain_names: List[str],
        alpha: float = 0.9,
        warmup_steps: int = 100,
        initial_weights: Optional[List[float]] = None,
    ):
        self.names  = domain_names
        self.K      = len(domain_names)
        self.alpha  = alpha
        self.warmup = warmup_steps
        self.step   = 0

        # Estimated importance-weighted reward per domain (init = 0)
        self.R_hat = np.zeros(self.K, dtype=np.float64)

        # Current mixing distribution
        if initial_weights is not None:
            w = np.array(initial_weights, dtype=np.float64)
            self.pi = w / w.sum()
        else:
            self.pi = np.ones(self.K, dtype=np.float64) / self.K

        # Per-step history for logging / analysis
        self.history: List[Dict] = []

    # ── core ODM equations ───────────────────────────────────────────────────

    def _exploration_rate(self) -> float:
        """E_t = min(1/K, sqrt(ln K / (K * t)))"""
        # FIX #4: log(1) == 0 when K==1, making E permanently 0.
        #         Return 1.0 (full uniform) — exploration is meaningless with 1 domain.
        if self.K == 1:
            return 1.0
        t = max(self.step, 1)
        return min(1.0 / self.K, math.sqrt(math.log(self.K) / (self.K * t)))

    def _compute_pi(self, E: float) -> np.ndarray:
        """
        π(D_i) = (1 - K*E) * softmax(E_{t-1} * R̂)[i]  +  E
        Gibbs distribution mixed with a uniform distribution.
        """
        # numerical stability: subtract max before exp
        scores  = (E * self.R_hat) - (E * self.R_hat).max()
        gibbs   = np.exp(scores)
        gibbs  /= gibbs.sum()
        new_pi  = (1.0 - self.K * E) * gibbs + E
        # clamp + renormalise for floating-point safety
        new_pi  = np.clip(new_pi, 0.0, 1.0)
        new_pi /= new_pi.sum()
        return new_pi

    # ── public API ───────────────────────────────────────────────────────────

    def sample_domain(self) -> int:
        """Sample a domain index according to the current π."""
        return int(np.random.choice(self.K, p=self.pi))

    def update(self, domain_idx: int, loss: float):
        """
        Called once per training step after the batch loss is computed.
        Updates the reward estimate for the sampled domain and recomputes π.

        Parameters
        ----------
        domain_idx : int   Index of the domain that was just sampled.
        loss       : float Scalar batch loss for that domain.
        """
        self.step += 1

        # Importance-weighted reward:  R_i = loss / π(D_i)
        reward = loss / (self.pi[domain_idx] + 1e-8)

        # Moving-average update (only for the domain we actually sampled)
        self.R_hat[domain_idx] = (
            self.alpha * self.R_hat[domain_idx] + (1.0 - self.alpha) * reward
        )

        # During warmup keep π uniform; afterwards apply the Exp3 policy
        if self.step <= self.warmup:
            new_pi = np.ones(self.K) / self.K
        else:
            E      = self._exploration_rate()
            new_pi = self._compute_pi(E)

        self.history.append({
            "step":   self.step,
            "domain": self.names[domain_idx],
            "loss":   round(loss, 6),
            "reward": round(reward, 6),
            "pi":     new_pi.tolist(),
            "R_hat":  self.R_hat.tolist(),
        })

        self.pi = new_pi

    def log_step(self):
        """Pretty-print current mixing weights."""
        weights_str = "  ".join(
            f"{n}={p:.3f}" for n, p in zip(self.names, self.pi)
        )
        print(f"  [ODM step {self.step}] π → {weights_str}")

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "domain_names": self.names,
                "alpha":        self.alpha,
                "warmup":       self.warmup,
                "final_pi":     self.pi.tolist(),
                "R_hat":        self.R_hat.tolist(),
                "history":      self.history,
            }, f, indent=2)
        print(f"ODM state saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Domain DataLoader wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MultiDomainIterator:
    """
    Wraps one DataLoader per domain.  Call .next(domain_idx) to get the
    next batch from that domain (automatically cycles when exhausted).
    """

    def __init__(self, loaders: List[DataLoader]):
        self.loaders = loaders
        self._iters  = [iter(dl) for dl in loaders]

    def next(self, domain_idx: int) -> Dict[str, torch.Tensor]:
        try:
            return next(self._iters[domain_idx])
        except StopIteration:
            # restart the exhausted domain's iterator
            self._iters[domain_idx] = iter(self.loaders[domain_idx])
            return next(self._iters[domain_idx])

    def __len__(self):
        """Total batches across all domains."""
        return sum(len(dl) for dl in self.loaders)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def forward_pass(model, batch: Dict, device: torch.device, use_amp: bool):
    """
    Single forward pass; returns (total_loss, mlm_loss, edge_loss).

    FIX #1: autocast lives here only.  The training loop must NOT open a
    second autocast context around this call.
    """
    if use_amp:
        dtype = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=dtype):
            out = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                position_ids   = batch["position_ids"],
                labels         = batch["labels"],
                edge_batch_idx = batch["edge_batch_idx"],
                edge_node1_pos = batch["edge_node1_pos"],
                edge_node2_pos = batch["edge_node2_pos"],
                edge_labels    = batch["edge_labels"],
            )
    else:
        out = model(
            input_ids      = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            position_ids   = batch["position_ids"],
            labels         = batch["labels"],
            edge_batch_idx = batch["edge_batch_idx"],
            edge_node1_pos = batch["edge_node1_pos"],
            edge_node2_pos = batch["edge_node2_pos"],
            edge_labels    = batch["edge_labels"],
        )
    return out["loss"], out["mlm_loss"], out["edge_loss"]


def clear_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def build_domain_loaders(
    domain_files: List[str],
    tokenizer,
    batch_size: int,
    max_length: int,
    val_split: float,
    collator,
    device: torch.device,                        # FIX #6: needed to gate pin_memory
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Build one train DataLoader and one val DataLoader per domain file.
    Returns (train_loaders, val_loaders).
    """
    # FIX #6: pin_memory only works correctly on CUDA; causes issues on MPS/CPU
    pin = device.type == "cuda"

    train_loaders, val_loaders = [], []

    for path in domain_files:
        ds      = GraphCodeBERTDataset(path, tokenizer, max_length)
        n_val   = max(1, int(val_split * len(ds)))
        n_train = len(ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_loaders.append(DataLoader(
            train_ds,
            batch_size  = batch_size,
            shuffle     = True,
            collate_fn  = collator,
            num_workers = min(4, os.cpu_count() or 1),
            pin_memory  = pin,
        ))
        val_loaders.append(DataLoader(
            val_ds,
            batch_size  = batch_size * 2,
            shuffle     = False,
            collate_fn  = collator,
            num_workers = min(4, os.cpu_count() or 1),
            pin_memory  = pin,
        ))

    return train_loaders, val_loaders


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_all_domains(
    model,
    val_loaders: List[DataLoader],
    domain_names: List[str],
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """Validate on every domain; return per-domain average loss."""
    model.eval()
    results = {}

    with torch.no_grad():
        for name, loader in zip(domain_names, val_loaders):
            total, count = 0.0, 0
            for batch in tqdm(loader, desc=f"Val [{name}]", leave=False):
                batch = move_batch(batch, device)
                loss, _, _ = forward_pass(model, batch, device, use_amp)
                total += loss.item()
                count += 1
            results[name] = total / max(count, 1)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main ODM Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def odm_train(args, config: Dict):
    """Full ODM training run."""

    set_seed(42)
    device, use_amp = setup_device()
    print(f"Device: {device}  |  AMP: {use_amp}")

    project_root = find_project_root()
    output_dir   = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── tokenizer & model ────────────────────────────────────────────────────
    checkpoint = getattr(args, "checkpoint_path", None)
    if checkpoint:
        print(f"Loading from checkpoint: {checkpoint}")
        tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
        model     = GraphCodeBERTWithEdgePrediction.from_pretrained(checkpoint).to(device)
    else:
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
        model     = GraphCodeBERTWithEdgePrediction("microsoft/graphcodebert-base").to(device)

    # ── domain data files ────────────────────────────────────────────────────
    domain_files = []
    domain_names = []

    cpp_file = getattr(args, "cpp_file", None) or config.get("data", {}).get("cpp_file")
    erl_file = getattr(args, "erl_file", None) or config.get("data", {}).get("erlang_file")

    if cpp_file and Path(project_root / cpp_file).exists():
        domain_files.append(str(project_root / cpp_file))
        domain_names.append("cpp")
    else:
        print(f"WARNING: C++ file not found ({cpp_file}), skipping domain.")

    if erl_file and Path(project_root / erl_file).exists():
        domain_files.append(str(project_root / erl_file))
        domain_names.append("erlang")
    else:
        print(f"WARNING: Erlang file not found ({erl_file}), skipping domain.")

    # To add more domains uncomment and extend:
    # extra_files = config.get("data", {}).get("extra_files", [])
    # for entry in extra_files:
    #     domain_files.append(str(project_root / entry["path"]))
    #     domain_names.append(entry["name"])

    if len(domain_files) < 1:
        raise ValueError("No domain data files found. Check your config.json paths.")
    if len(domain_files) == 1:
        print("WARNING: Only one domain found — ODM reduces to standard training.")

    print(f"\nDomains ({len(domain_names)}): {domain_names}")
    for n, p in zip(domain_names, domain_files):
        print(f"  {n}: {p}")

    # ── training hyper-params ────────────────────────────────────────────────
    train_cfg    = config.get("train", {})
    batch_size   = getattr(args, "batch_size",       None) or train_cfg.get("batch_size",       8)
    epochs       = getattr(args, "epochs",           None) or train_cfg.get("epochs",           5)
    lr           = getattr(args, "learning_rate",    None) or train_cfg.get("learning_rate",    2e-5)
    max_length   = getattr(args, "max_length",       None) or train_cfg.get("max_length",       512)
    warmup_steps = getattr(args, "warmup_steps",     None) or train_cfg.get("warmup_steps",     500)
    mlm_prob     = getattr(args, "mlm_probability",  None) or train_cfg.get("mlm_probability",  0.15)
    val_split    = getattr(args, "validation_split", None) or train_cfg.get("validation_split", 0.1)
    weight_decay = getattr(args, "weight_decay",     None) or train_cfg.get("weight_decay",     0.01)
    patience     = getattr(args, "early_stopping_patience", None) or train_cfg.get("early_stopping_patience", 3)

    # FIX #3: all ODM params use `or` so that None from argparse (default=None)
    #         correctly falls through to the computed default expression.
    odm_alpha    = getattr(args, "odm_alpha",        None) or 0.9
    odm_warmup   = getattr(args, "odm_warmup_steps", None) or max(50, warmup_steps // 10)
    log_interval = getattr(args, "log_interval",     None) or 50

    print(f"\nTraining config:")
    print(f"  batch_size={batch_size}  epochs={epochs}  lr={lr}  max_length={max_length}")
    print(f"  warmup_steps={warmup_steps}  mlm_prob={mlm_prob}  val_split={val_split}")
    print(f"  odm_alpha={odm_alpha}  odm_warmup={odm_warmup}  log_interval={log_interval}")

    # ── data loaders ─────────────────────────────────────────────────────────
    collator = MLMWithEdgePredictionCollator(tokenizer, mlm_probability=mlm_prob)

    print("\nBuilding domain DataLoaders...")
    train_loaders, val_loaders = build_domain_loaders(
        domain_files, tokenizer, batch_size, max_length, val_split, collator,
        device,                                  # FIX #6
    )

    # Steps per epoch ≈ average loader length; used for the LR scheduler
    avg_domain_steps = int(np.mean([len(dl) for dl in train_loaders]))
    total_steps      = avg_domain_steps * epochs
    print(f"Approx training steps: {total_steps}  ({avg_domain_steps}/epoch × {epochs} epochs)")

    # ── optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )
    scaler = GradScaler() if use_amp else None

    # ── ODM bandit ────────────────────────────────────────────────────────────
    bandit = ODMBandit(
        domain_names = domain_names,
        alpha        = odm_alpha,
        warmup_steps = odm_warmup,
    )

    # ── per-domain iterators (each cycles independently) ─────────────────────
    multi_iter = MultiDomainIterator(train_loaders)

    # ── bookkeeping ───────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    best_model_dir   = output_dir / "best_model"
    checkpoints_dir  = output_dir / "checkpoints"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    # FIX #7: initialise epoch so the post-loop summary never raises NameError
    #         if domain_files was empty and the loop never ran (we raise above,
    #         but this is defensive).
    epoch       = -1
    global_step = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Epoch loop
    # ─────────────────────────────────────────────────────────────────────────
    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")
        clear_cache(device)

        model.train()

        domain_losses  = {n: [] for n in domain_names}
        epoch_loss_sum = 0.0
        epoch_steps    = 0

        progress = tqdm(range(avg_domain_steps), desc=f"Epoch {epoch+1}")

        for _ in progress:
            optimizer.zero_grad()

            # ── ODM: sample domain & fetch batch ───────────────────────────
            d_idx = bandit.sample_domain()
            batch = move_batch(multi_iter.next(d_idx), device)

            # ── forward + backward ─────────────────────────────────────────
            # FIX #1: forward_pass owns autocast — no second context here.
            # FIX #2: scaler.unscale_ must precede clip_grad_norm_ under AMP.
            loss, mlm_loss, edge_loss = forward_pass(model, batch, device, use_amp)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)                      # FIX #2
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            loss_val = loss.item()

            # ── ODM: update bandit weights ─────────────────────────────────
            bandit.update(d_idx, loss_val)

            # ── bookkeeping ────────────────────────────────────────────────
            domain_losses[domain_names[d_idx]].append(loss_val)
            epoch_loss_sum += loss_val
            epoch_steps    += 1
            global_step    += 1

            avg    = epoch_loss_sum / epoch_steps
            pi_str = " ".join(
                f"{n[0]}={p:.2f}" for n, p in zip(domain_names, bandit.pi)
            )
            progress.set_postfix({
                "loss": f"{loss_val:.4f}",
                "avg":  f"{avg:.4f}",
                "π":    pi_str,
                "lr":   f"{optimizer.param_groups[0]['lr']:.2e}",
            })

            if global_step % log_interval == 0:
                bandit.log_step()

            clear_cache(device)

        # ── per-epoch train summary ───────────────────────────────────────
        print(f"\nEpoch {epoch+1} Train Summary:")
        for name in domain_names:
            dl  = domain_losses[name]
            idx = domain_names.index(name)
            if dl:
                print(f"  {name:12s}  avg_loss={np.mean(dl):.6f}  "
                      f"n_batches={len(dl)}  π={bandit.pi[idx]:.4f}")
            else:
                print(f"  {name:12s}  (no batches sampled this epoch)  "
                      f"π={bandit.pi[idx]:.4f}")

        # ── validation ────────────────────────────────────────────────────
        clear_cache(device)
        val_results  = validate_all_domains(model, val_loaders, domain_names, device, use_amp)
        avg_val_loss = float(np.mean(list(val_results.values())))

        print(f"\nEpoch {epoch+1} Validation:")
        for name, vloss in val_results.items():
            print(f"  {name:12s}  val_loss={vloss:.6f}")
        print(f"  {'AVERAGE':12s}  val_loss={avg_val_loss:.6f}")

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"  Peak GPU memory: {peak_mem:.2f} GB")

        # ── checkpoint every epoch (model + full training state) ──────────
        ckpt_dir = checkpoints_dir / f"epoch_{epoch+1:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))

        # Save optimizer / scheduler / scaler / bandit so training can be
        # resumed from any epoch without restarting from scratch.
        torch.save({
            "epoch":                epoch + 1,
            "global_step":          global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict() if scaler else None,
            "best_val_loss":        best_val_loss,
            "bandit_pi":            bandit.pi.tolist(),
            "bandit_R_hat":         bandit.R_hat.tolist(),
            "bandit_step":          bandit.step,
        }, ckpt_dir / "training_state.pt")

        print(f"Checkpoint saved → {ckpt_dir}")

        # ── best model ────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            model.save_pretrained(str(best_model_dir))
            tokenizer.save_pretrained(str(best_model_dir))
            print(f"★ New best model (val={best_val_loss:.6f}) → {best_model_dir}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{patience}")

        # ── metrics CSV row ───────────────────────────────────────────────
        row = {
            "epoch":          epoch + 1,
            "avg_train_loss": epoch_loss_sum / max(epoch_steps, 1),
            "avg_val_loss":   avg_val_loss,
            "best_val_loss":  best_val_loss,
            "lr":             optimizer.param_groups[0]["lr"],
        }
        for name in domain_names:
            idx = domain_names.index(name)
            row[f"train_loss_{name}"] = (
                float(np.mean(domain_losses[name])) if domain_losses[name] else None
            )
            row[f"val_loss_{name}"] = val_results.get(name)
            row[f"pi_{name}"]       = float(bandit.pi[idx])
        metrics_rows.append(row)

        # ── early stopping ────────────────────────────────────────────────
        if patience_counter >= patience:
            print(f"\nEarly stopping — no improvement for {patience} epochs.")
            break

    # ── post-training saves ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Training complete. Saving artefacts...")

    # ODM bandit state (full sampling history, final π, R̂)
    bandit.save(str(output_dir / "odm_state.json"))

    # Metrics CSV
    csv_path = output_dir / "training_metrics.csv"
    if metrics_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_rows[0].keys())
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"Metrics CSV → {csv_path}")

    # Summary JSON
    # FIX #7: epoch is always defined (initialised to -1 above loop)
    summary = {
        "best_val_loss": best_val_loss,
        "total_epochs":  epoch + 1,
        "global_steps":  global_step,
        "domain_names":  domain_names,
        "final_pi":      bandit.pi.tolist(),
        "odm_alpha":     odm_alpha,
        "odm_warmup":    odm_warmup,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest model      → {best_model_dir}")
    print(f"Final π weights → {dict(zip(domain_names, [round(p, 4) for p in bandit.pi]))}")
    print(f"All outputs     → {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GraphCodeBERT + Online Data Mixing (ODM) training"
    )

    # Data
    parser.add_argument("--cpp_file", type=str, default=None,
                        help="Path to C++ JSONL (relative to project root; falls back to config.json)")
    parser.add_argument("--erl_file", type=str, default=None,
                        help="Path to Erlang JSONL (relative to project root; falls back to config.json)")

    # Model
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Resume from existing checkpoint directory")
    parser.add_argument("--output_dir",      type=str, default=None,
                        help="Output directory (relative to project root)")

    # Training hypers — all default to None so config.json values take effect
    parser.add_argument("--batch_size",              type=int,   default=None)
    parser.add_argument("--epochs",                  type=int,   default=None)
    parser.add_argument("--learning_rate",           type=float, default=None)
    parser.add_argument("--max_length",              type=int,   default=None)
    parser.add_argument("--warmup_steps",            type=int,   default=None)
    parser.add_argument("--mlm_probability",         type=float, default=None)
    parser.add_argument("--validation_split",        type=float, default=None)
    parser.add_argument("--weight_decay",            type=float, default=None)
    parser.add_argument("--early_stopping_patience", type=int,   default=None)

    # ODM-specific — FIX #3: default=None everywhere so `or` fallback works
    parser.add_argument("--odm_alpha",        type=float, default=None,
                        help="Moving-average decay for reward estimates (default 0.9)")
    parser.add_argument("--odm_warmup_steps", type=int,   default=None,
                        help="Steps before ODM starts adapting weights (default: warmup_steps/10)")
    parser.add_argument("--log_interval",     type=int,   default=None,
                        help="Print ODM π every N steps (default 50)")

    args = parser.parse_args()

    config = load_config()
    if args.output_dir is None:
        args.output_dir = config.get("train", {}).get("output_dir", "runs/odm")

    odm_train(args, config)


if __name__ == "__main__":
    main()