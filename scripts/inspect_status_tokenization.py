#!/usr/bin/env python
"""Inspect how Normal/Abnormal/prompt tokenize for Stage-1 generation.

Server usage:
    python scripts/inspect_status_tokenization.py \
        --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct
"""
import argparse
import sys

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--normal-answer", default="Normal")
    parser.add_argument("--abnormal-answer", default="Abnormal")
    parser.add_argument("--status-prompt", default="Current video status:")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    prompt_ids = tokenizer.encode(args.status_prompt, add_special_tokens=False)
    normal_ids = tokenizer.encode(args.normal_answer, add_special_tokens=False)
    abnormal_ids = tokenizer.encode(args.abnormal_answer, add_special_tokens=False)

    eos_id = tokenizer.eos_token_id
    print(f"prompt   '{args.status_prompt}'")
    print(f"  token IDs: {prompt_ids}")
    print(f"  decoded:   '{tokenizer.decode(prompt_ids)}'")
    print()
    print(f"Normal   '{args.normal_answer}'")
    print(f"  token IDs: {normal_ids}")
    print(f"  decoded:   '{tokenizer.decode(normal_ids)}'")
    print()
    print(f"Abnormal '{args.abnormal_answer}'")
    print(f"  token IDs: {abnormal_ids}")
    print(f"  decoded:   '{tokenizer.decode(abnormal_ids)}'")
    print()
    print(f"eos token ID: {eos_id}")
    print(f"vocab size:   {tokenizer.vocab_size}")

    # assertions
    assert len(normal_ids) >= 1, "Normal answer tokenized to empty"
    assert len(abnormal_ids) >= 1, "Abnormal answer tokenized to empty"
    assert normal_ids != abnormal_ids, "Normal and Abnormal must differ"
    assert tokenizer.decode(normal_ids) == args.normal_answer or \
           tokenizer.decode(normal_ids).strip() == args.normal_answer, \
           f"Normal round-trip failed: '{tokenizer.decode(normal_ids)}'"
    assert tokenizer.decode(abnormal_ids) == args.abnormal_answer or \
           tokenizer.decode(abnormal_ids).strip() == args.abnormal_answer, \
           f"Abnormal round-trip failed: '{tokenizer.decode(abnormal_ids)}'"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
