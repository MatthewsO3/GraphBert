"""
merge_and_retokenize.py
=======================
1. Load C++ training data  (already uses RoBERTa BPE tokens — left untouched)
2. Load Erlang training data (uses naive whitespace/punct tokens)
3. Re-tokenize every Erlang sample with the RoBERTa BPE tokenizer
4. Realign  variable_positions  and  dataflow_graph  position indices so they
   point into the *new* BPE token list instead of the old naive token list
5. Shuffle C++ + Erlang together and write a mixed JSONL file

Position-realignment strategy
------------------------------
The naive tokenizer splits on whitespace/punctuation, so each naive token is
a contiguous substring of the raw `code`.  We:
  a) Rebuild the character offsets of every naive token by scanning `code`
     left-to-right (matching each token as a literal substring).
  b) Tokenize `code` with RoBERTa using `return_offsets_mapping=True` so we
     get the char-span [start, end) for every BPE token.
  c) For each naive position that needs mapping we look up its char-span and
     find the *first* BPE token whose span overlaps it.

This is O(n) per sample and handles multi-char and Unicode tokens correctly.
"""

import json
import random
from typing import Optional
from transformers import RobertaTokenizerFast   # Fast variant gives offset maps

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
TOKENIZER_NAME = "microsoft/graphcodebert-base"
print(f"Loading tokenizer: {TOKENIZER_NAME}  ...")
tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_NAME)
print("Tokenizer ready.\n")


# ---------------------------------------------------------------------------
# Core: tokenize + build offset map
# ---------------------------------------------------------------------------

def bpe_tokens_and_offsets(code: str):
    """
    Returns
      tokens   : list[str]  - BPE token strings  (G / C / t prefixes kept)
      offsets  : list[tuple[int,int]]  - (char_start, char_end) per token
                 end is exclusive;  (0, 0) for special tokens (none here
                 because add_special_tokens=False)
    """
    enc = tokenizer(
        code,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    tokens  = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    offsets = enc["offset_mapping"]          # list of [start, end] pairs
    return tokens, offsets


# ---------------------------------------------------------------------------
# Core: map naive token positions to BPE token positions
# ---------------------------------------------------------------------------

def build_naive_char_offsets(code: str, naive_tokens: list) -> list:
    """
    Reconstruct the character offsets of each naive token by scanning `code`
    left-to-right and matching each token as a literal substring.

    Returns a list of (start, end) tuples parallel to `naive_tokens`.
    Tokens that cannot be located (e.g. empty strings) get offset (-1, -1).
    """
    offsets = []
    cursor  = 0
    for tok in naive_tokens:
        if not tok:
            offsets.append((-1, -1))
            continue
        idx = code.find(tok, cursor)
        if idx == -1:
            # Fallback: search from 0 (handles rare out-of-order edge cases)
            idx = code.find(tok)
        if idx == -1:
            offsets.append((-1, -1))
        else:
            offsets.append((idx, idx + len(tok)))
            cursor = idx + len(tok)
    return offsets


def build_naive_to_bpe_map(
    naive_offsets: list,
    bpe_offsets:   list,
) -> dict:
    """
    For every naive token index i, find the index of the *first* BPE token
    whose character span overlaps naive_offsets[i].

    Overlap condition:  bpe_start < naive_end  AND  bpe_end > naive_start
    (standard interval overlap, both ends exclusive on the right)

    Returns a dict  { naive_index : bpe_index }.
    Naive tokens with offset (-1,-1) are omitted.
    """
    mapping = {}

    for ni, (ns, ne) in enumerate(naive_offsets):
        if ns == -1:
            continue
        # Linear scan is fine; samples are small (< a few hundred tokens).
        # For very large files a binary search on bpe_offsets would be faster.
        for bi, (bs, be) in enumerate(bpe_offsets):
            if bs < ne and be > ns:   # overlap
                mapping[ni] = bi
                break

    return mapping


# ---------------------------------------------------------------------------
# Core: realign one sample
# ---------------------------------------------------------------------------

def realign_sample(sample: dict) -> dict:
    """
    Given an Erlang sample with naive code_tokens / variable_positions /
    dataflow_graph, return a new sample where:
      - code_tokens        is replaced with RoBERTa BPE tokens
      - variable_positions position indices are remapped
      - dataflow_graph     position indices (col 1 and inside col 4) are remapped

    Any position that cannot be mapped is left as-is with a warning flag so
    downstream tools can decide how to handle it.
    """
    sample   = dict(sample)   # shallow copy
    code     = sample.get("code", "")

    if not code:
        return sample

    naive_tokens = sample.get("code_tokens", [])

    # 1. BPE tokenize
    bpe_tokens, bpe_offsets = bpe_tokens_and_offsets(code)
    sample["code_tokens"] = bpe_tokens

    if not naive_tokens:
        return sample

    # 2. Build char offsets for naive tokens
    naive_offsets = build_naive_char_offsets(code, naive_tokens)

    # 3. Build naive->BPE index map
    n2b = build_naive_to_bpe_map(naive_offsets, bpe_offsets)

    def remap(pos: int) -> int:
        """Remap a naive token index to a BPE token index. -1 = unmappable."""
        return n2b.get(pos, -1)

    # 4. Realign variable_positions
    #    Format: [[naive_pos, var_name, is_def], ...]
    new_var_pos = []
    for entry in sample.get("variable_positions", []):
        entry      = list(entry)
        new_pos    = remap(entry[0])
        entry[0]   = new_pos if new_pos != -1 else entry[0]
        if new_pos == -1:
            entry.append("__pos_unmapped__")   # flag for inspection
        new_var_pos.append(entry)
    sample["variable_positions"] = new_var_pos

    # 5. Realign dataflow_graph
    #    Format: [[var_name, naive_pos, rel, [src_names], [src_naive_positions]], ...]
    new_dfg = []
    for edge in sample.get("dataflow_graph", []):
        edge         = list(edge)
        # col 1: position of this node
        new_pos      = remap(edge[1])
        edge[1]      = new_pos if new_pos != -1 else edge[1]
        # col 4: list of source positions
        src_positions = edge[4] if len(edge) > 4 else []
        new_src       = []
        for sp in src_positions:
            nb = remap(sp)
            new_src.append(nb if nb != -1 else sp)
        if len(edge) > 4:
            edge[4] = new_src
        new_dfg.append(edge)
    sample["dataflow_graph"] = new_dfg

    return sample


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save_jsonl(samples: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Sanity-check helper  (run on a single sample to verify alignment)
# ---------------------------------------------------------------------------

def verify_sample(original: dict, realigned: dict) -> None:
    """
    Print a side-by-side view of variable_positions before and after so you
    can manually verify the mapping looks correct.
    """
    print("\n-- Verification --------------------------------------------------")
    old_tokens = original.get("code_tokens", [])
    new_tokens = realigned.get("code_tokens", [])
    print(f"  Naive tokens : {len(old_tokens)}")
    print(f"  BPE   tokens : {len(new_tokens)}")
    print()
    for old_entry, new_entry in zip(
        original.get("variable_positions", []),
        realigned.get("variable_positions", []),
    ):
        old_pos  = old_entry[0]
        new_pos  = new_entry[0]
        old_tok  = old_tokens[old_pos] if 0 <= old_pos < len(old_tokens) else "?"
        new_tok  = new_tokens[new_pos] if 0 <= new_pos < len(new_tokens) else "?"
        var_name = old_entry[1]
        flag     = "  WARNING: unmapped" if "__pos_unmapped__" in new_entry else ""
        print(
            f"  {var_name:<12}  naive[{old_pos:3d}]={old_tok!r:<22} "
            f"->  bpe[{new_pos:3d}]={new_tok!r}{flag}"
        )
    print("------------------------------------------------------------------\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def retokenize_erlang_only(
    erlang_file:      str,
    output_file:      str,
    max_samples:      Optional[int] = None,
    seed:             int = 42,
    verify_first_n:   int = 5,
) -> None:
    rng = random.Random(seed)

    # -- Load Erlang ---------------------------------------------------------
    erlang_raw = load_jsonl(erlang_file)
    if max_samples and len(erlang_raw) > max_samples:
        erlang_raw = rng.sample(erlang_raw, max_samples)
        print(f"Erlang : sampled {max_samples:,} from {erlang_file}")
    else:
        print(f"Erlang : loaded  {len(erlang_raw):,} samples from {erlang_file}")

    # -- Retokenize & Realign ------------------------------------------------
    print("Retokenizing & realigning Erlang samples ...")
    erlang_retokenized = []
    unmapped_total = 0

    for i, raw in enumerate(erlang_raw):
        realigned = realign_sample(raw)
        erlang_retokenized.append(realigned)

        # Summary for unmapped positions
        for entry in realigned.get("variable_positions", []):
            if "__pos_unmapped__" in entry:
                unmapped_total += 1

        # Print verification for the first few
        if i < verify_first_n:
            print(f"\n[Sample {i}]  idx={raw.get('idx', '?')}")
            verify_sample(raw, realigned)

        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1:,} / {len(erlang_raw):,} done")

    # -- Save Output ---------------------------------------------------------
    save_jsonl(erlang_retokenized, output_file)
    
    print(f"\nRetokenization complete.")
    print(f"Unmapped positions total: {unmapped_total}")
    print(f"Final file: {output_file} ({len(erlang_retokenized):,} samples)")


if __name__ == "__main__":
    # Update these paths for your Erlang-only run
    retokenize_erlang_only(
        erlang_file      = "/home/mczap/erlangbert/erlang_corpus_scraper/output/graphcodebert_data/full.jsonl",
        output_file      = "/home/mczap/GraphBert/GraphBERT/data/erlang/retokenized_train.jsonl",
        max_samples      = None,  # Use all samples
        verify_first_n   = 5      # Keep this on to ensure the first 5 look good
    )