import argparse
import os
from typing import Dict, Tuple

import numpy as np


def read_item2id(path: str) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid line {line_num} in {path}: expected 2 tab-separated fields")
            original_raw, mapped_raw = parts
            try:
                original_idx = int(original_raw)
            except ValueError as e:
                raise ValueError(f"Invalid original item id at line {line_num} in {path}: {original_raw!r}") from e
            try:
                mapped_idx = int(mapped_raw)
            except ValueError as e:
                raise ValueError(f"Invalid mapped item id at line {line_num} in {path}: {mapped_raw!r}") from e

            mapping[original_idx] = mapped_idx
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap an embedding matrix using a Merlin .item2id file")
    parser.add_argument("--dataset", type=str, default="merlin", help="Dataset name/prefix for mapping file")
    parser.add_argument("--root", type=str, default="./data/merlin/merlin", help="Dataset directory containing *.item2id")
    parser.add_argument(
        "--embedding_input_source",
        type=str,
        required=True,
        help="Path to source embedding .npy (row index equals original item id from parquet)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output .npy path (default: <root>/<dataset>_emb.npy)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any mapping points outside embedding_input_source",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    item2id_path = os.path.join(args.root, f"{args.dataset}.item2id")
    mapping = read_item2id(item2id_path)
    if not mapping:
        raise ValueError(f"No mappings found in {item2id_path}")

    src = np.load(args.embedding_input_source, mmap_mode="r")
    if src.ndim < 2:
        raise ValueError(f"Expected embedding_input_source to be at least 2D, got shape={src.shape}")

    max_mapped = max(mapping.values())
    out_shape: Tuple[int, ...] = (max_mapped + 1, *src.shape[1:])
    out = np.zeros(out_shape, dtype=src.dtype)

    bad = 0
    for original_idx, mapped_idx in mapping.items():
        if original_idx < 0 or original_idx >= src.shape[0]:
            bad += 1
            if args.strict:
                raise IndexError(
                    f"original_idx {original_idx} out of range for embedding_input_source rows={src.shape[0]}"
                )
            continue
        out[mapped_idx] = src[original_idx]

    output_file = args.output_file or os.path.join(args.root, f"{args.dataset}_emb.npy")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    np.save(output_file, out)

    print("===================================================")
    print(" Merlin Embedding Remap Completed Successfully")
    print("===================================================")
    print(f"item2id: {item2id_path}")
    print(f"src: {args.embedding_input_source} shape={tuple(src.shape)}")
    print(f"out: {output_file} shape={tuple(out.shape)}")
    if bad:
        print(f"Warnings: skipped {bad} out-of-range mappings (use --strict to fail).")


if __name__ == "__main__":
    main()

