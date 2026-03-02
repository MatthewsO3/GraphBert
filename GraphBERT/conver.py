#!/usr/bin/env python3
"""
convert_cpp_to_erlang_format.py

Converts a C++ trained model checkpoint to Erlang format.

The Erlang script expects the checkpoint to have a custom format with:
  model.bin containing: {'model_state_dict': {...}, 'config': {...}}

The C++ script saves in HuggingFace format:
  pytorch_model.bin containing just the state dict

This script converts between the two formats.

Usage:
    python convert_cpp_to_erlang_format.py \
      --cpp_checkpoint ./cpp_results/best_model/ \
      --output_dir ./cpp_trained_for_erlang/
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import RobertaForMaskedLM, RobertaTokenizer


def convert_cpp_to_erlang_format(cpp_checkpoint_path: str, output_dir: str):
    """
    Convert C++ checkpoint to Erlang format.
    
    Args:
        cpp_checkpoint_path: Path to C++ best_model/ or checkpoint
        output_dir: Where to save the converted checkpoint
    """
    
    cpp_path = Path(cpp_checkpoint_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Converting C++ Model to Erlang Format")
    print("=" * 80)
    
    # Step 1: Verify input checkpoint
    print(f"\n1. Verifying C++ checkpoint at: {cpp_path}")
    
    # Check for pytorch_model.bin (HuggingFace format)
    pytorch_model_path = cpp_path / "pytorch_model.bin"
    if not pytorch_model_path.exists():
        print(f"   ⚠️  pytorch_model.bin not found")
        print(f"   Looking for model.safetensors instead...")
        safetensors_path = cpp_path / "model.safetensors"
        if safetensors_path.exists():
            pytorch_model_path = safetensors_path
            print(f"   ✓ Found model.safetensors")
        else:
            raise FileNotFoundError(
                f"No pytorch_model.bin or model.safetensors in {cpp_path}\n"
                f"Available files: {list(cpp_path.glob('*'))}"
            )
    else:
        print(f"   ✓ Found pytorch_model.bin")
    
    # Check for config.json
    config_path = cpp_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {cpp_path}")
    print(f"   ✓ Found config.json")
    
    # Step 2: Load the model and config
    print(f"\n2. Loading model and config...")
    try:
        model = RobertaForMaskedLM.from_pretrained(str(cpp_path))
        print(f"   ✓ Model loaded")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        raise
    
    # Step 3: Extract state dict
    print(f"\n3. Extracting state dict...")
    state_dict = model.state_dict()
    print(f"   ✓ State dict extracted ({len(state_dict)} parameters)")
    
    # Step 4: Create Erlang format checkpoint
    print(f"\n4. Creating Erlang format checkpoint...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create custom checkpoint format
    erlang_checkpoint = {
        'model_state_dict': state_dict,
        'config': config
    }
    
    # Save as model.bin
    output_model_path = output_path / "model.bin"
    torch.save(erlang_checkpoint, output_model_path)
    print(f"   ✓ Saved Erlang format checkpoint to {output_model_path}")
    print(f"     File size: {output_model_path.stat().st_size / 1e9:.2f} GB")
    
    # Step 5: Copy tokenizer files
    print(f"\n5. Copying tokenizer files...")
    
    tokenizer_files = [
        'vocab.json',
        'merges.txt',
        'tokenizer_config.json',
        'special_tokens_map.json',
    ]
    
    for filename in tokenizer_files:
        src = cpp_path / filename
        dst = output_path / filename
        if src.exists():
            with open(src, 'r') as f_src, open(dst, 'w') as f_dst:
                f_dst.write(f_src.read())
            print(f"   ✓ Copied {filename}")
        else:
            print(f"   ⚠️  {filename} not found")
    
    # Step 6: Save conversion metadata
    print(f"\n6. Saving conversion metadata...")
    
    metadata = {
        'source': str(cpp_path.absolute()),
        'conversion_script': 'convert_cpp_to_erlang_format.py',
        'original_format': 'C++ (HuggingFace)',
        'target_format': 'Erlang (Custom)',
        'config': config
    }
    
    with open(output_path / "conversion_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Saved conversion_metadata.json")
    
    print("\n" + "=" * 80)
    print("✅ Conversion Complete!")
    print("=" * 80)
    print(f"\nYou can now train with Erlang script:")
    print(f"  python erlang_train.py \\")
    print(f"    --checkpoint-path {output_path} \\")
    print(f"    --train-file data/erlang_functions.jsonl \\")
    print(f"    --output-dir erlang_results_continued \\")
    print(f"    --num-epochs 2")
    print()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert C++ trained model to Erlang format"
    )
    parser.add_argument(
        "--cpp_checkpoint",
        type=str,
        required=True,
        help="Path to C++ checkpoint (best_model or epoch checkpoint)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cpp_trained_for_erlang",
        help="Output directory for converted checkpoint"
    )
    
    args = parser.parse_args()
    
    print(f"Starting conversion...")
    print(f"Input: {args.cpp_checkpoint}")
    print(f"Output: {args.output_dir}")
    
    try:
        result = convert_cpp_to_erlang_format(args.cpp_checkpoint, args.output_dir)
        print(f"\n✅ Success! Model saved to: {result}")
        return 0
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)