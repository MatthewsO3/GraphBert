#!/usr/bin/env python3
"""
Analyze ODM state history to visualize when π diverged from uniform
and when domains' difficulties converged.
"""

import json
import sys
from pathlib import Path

def analyze_odm_state(odm_json_path: str):
    """Load odm_state.json and extract key milestones."""
    
    with open(odm_json_path) as f:
        data = json.load(f)
    
    history = data["history"]
    warmup = data["warmup"]
    
    print(f"ODM Training Analysis")
    print(f"=" * 70)
    print(f"Total steps: {len(history)}")
    print(f"Warmup steps: {warmup}")
    print(f"Domain names: {data['domain_names']}")
    print()
    
    # Track when π first diverges from [0.5, 0.5]
    first_diverge_step = None
    max_diverge_step = None
    max_diverge_amount = 0.0
    
    cpp_samples = 0
    erl_samples = 0
    
    for entry in history:
        step = entry["step"]
        domain = entry["domain"]
        pi = entry["pi"]
        R_hat = entry["R_hat"]
        loss = entry["loss"]
        
        if domain == "cpp":
            cpp_samples += 1
        else:
            erl_samples += 1
        
        # Detect divergence from [0.5, 0.5]
        divergence = abs(pi[0] - 0.5)
        if divergence > 0.001 and first_diverge_step is None:
            first_diverge_step = step
        
        if divergence > max_diverge_amount:
            max_diverge_amount = divergence
            max_diverge_step = step
    
    print(f"First divergence from [0.5, 0.5]: step {first_diverge_step}")
    print(f"Maximum divergence: step {max_diverge_step} → π = [0.5 ± {max_diverge_amount:.4f}]")
    print(f"Final π: {history[-1]['pi']}")
    print(f"Final R̂: {history[-1]['R_hat']}")
    print()
    
    print(f"Sample count: C++ = {cpp_samples}, Erlang = {erl_samples}")
    print(f"Ratio: C++/Erlang = {cpp_samples/erl_samples:.3f}")
    print()
    
    # Sample some key steps
    sample_steps = [
        warmup,                    # End of warmup
        warmup + 100,              # Just after warmup
        len(history) // 4,         # 25%
        len(history) // 2,         # 50% (mid-training)
        3 * len(history) // 4,     # 75%
        len(history) - 1,          # Final
    ]
    
    print("Snapshot at key milestones:")
    print("-" * 70)
    print(f"{'Step':>6} {'Domain':>8} {'Loss':>8} {'π[cpp]':>8} {'R̂[cpp]':>8}")
    print("-" * 70)
    
    for idx in sample_steps:
        if idx >= len(history):
            continue
        e = history[idx]
        print(f"{e['step']:6d} {e['domain']:>8} {e['loss']:8.4f} {e['pi'][0]:8.4f} {e['R_hat'][0]:8.4f}")
    
    print()
    print("Conclusion:")
    print("-" * 70)
    if first_diverge_step and first_diverge_step < warmup + 1000:
        print(f"✓ ODM adapted quickly (by step {first_diverge_step})")
    else:
        print(f"✗ ODM took a long time to adapt (step {first_diverge_step})")
    
    if max_diverge_amount > 0.05:
        print(f"✓ Significant preference detected (max divergence: {max_diverge_amount:.4f})")
    else:
        print(f"✗ Minimal preference (max divergence: {max_diverge_amount:.4f})")
    
    final_r_diff = abs(history[-1]["R_hat"][0] - history[-1]["R_hat"][1])
    if final_r_diff < 0.2:
        print(f"✓ Domains converged to similar difficulty (ΔR̂ = {final_r_diff:.4f})")
    else:
        print(f"✗ Domains remained different (ΔR̂ = {final_r_diff:.4f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_odm.py <path_to_odm_state.json>")
        sys.exit(1)
    
    analyze_odm_state(sys.argv[1])