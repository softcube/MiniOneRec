import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def check_path(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json_file(data: Any, file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_remap_index(index_map: Dict[str, int], file_path: str) -> None:
    with open(file_path, "w") as f:
        for original, mapped in index_map.items():
            f.write(f"{original}\t{mapped}\n")


def _is_nan(value: Any) -> bool:
    try:
        result = pd.isna(value)
        if isinstance(result, (bool, int)):
            return bool(result)
        # Non-scalar (e.g., ndarray/list) -> not a NaN marker for our purposes
        return False
    except Exception:
        return False


def _to_python_list(value: Any) -> List[Any]:
    if value is None or _is_nan(value):
        return []

    if isinstance(value, list):
        return value

    # pyarrow ListScalar / Array
    to_pylist = getattr(value, "to_pylist", None)
    if callable(to_pylist):
        return list(to_pylist())

    # numpy array / pandas array
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return list(tolist())

    # Some datasets store lists as JSON strings
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return [value]

    if isinstance(value, tuple):
        return list(value)

    # Fallback: try to iterate
    try:
        return list(value)  # type: ignore[arg-type]
    except Exception:
        return [value]


def _normalize_id(value: Any) -> Optional[str]:
    if value is None or _is_nan(value):
        return None
    return str(value)


def _ensure_columns(df: pd.DataFrame, path: str, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def _iter_sequences(
    df: pd.DataFrame,
    *,
    user_col: str,
    items_col: str,
) -> Iterable[Tuple[str, List[str]]]:
    for row in df.itertuples(index=False):
        user_raw = getattr(row, user_col)
        items_raw = getattr(row, items_col)

        user = _normalize_id(user_raw)
        if user is None:
            continue

        items = [_normalize_id(x) for x in _to_python_list(items_raw)]
        items = [x for x in items if x is not None]
        if len(items) < 2:
            continue

        yield user, items


def _assign_id(mapping: Dict[str, int], original: str) -> int:
    mapped = mapping.get(original)
    if mapped is None:
        mapped = len(mapping)
        mapping[original] = mapped
    return mapped


def _write_split_inter(
    *,
    df: pd.DataFrame,
    out_path: str,
    dataset: str,
    split: str,
    user2index: Dict[str, int],
    item2index: Dict[str, int],
    user_col: str,
    items_col: str,
    strategy: str,
    max_history_window: int,
    max_history_write: int,
) -> Dict[int, List[int]]:
    user2items: Dict[int, List[int]] = {}
    out_file = os.path.join(out_path, f"{dataset}.{split}.inter")

    with open(out_file, "w") as f:
        f.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")

        for user_raw, items_raw in _iter_sequences(df, user_col=user_col, items_col=items_col):
            uid = _assign_id(user2index, user_raw)

            item_ids_raw = [_assign_id(item2index, item) for item in items_raw]

            if strategy == "last":
                history = item_ids_raw[:-1]
                target = item_ids_raw[-1]
                history = history[-max_history_write:]
                f.write(f"{uid}\t{' '.join(map(str, history))}\t{target}\n")
                user2items.setdefault(uid, []).extend(item_ids_raw)
            elif strategy == "sliding":
                for i in range(1, len(item_ids_raw)):
                    start = max(i - max_history_window, 0)
                    history = item_ids_raw[start:i]
                    target = item_ids_raw[i]
                    history = history[-max_history_write:]
                    f.write(f"{uid}\t{' '.join(map(str, history))}\t{target}\n")
                user2items.setdefault(uid, []).extend(item_ids_raw)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

    return user2items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Merlin parquet splits to Amazon18-style RQ-ready files")
    parser.add_argument("--dataset", type=str, default="merlin", help="Dataset name/prefix for output files")
    parser.add_argument("--output_path", type=str, default="./data/merlin", help="Output root directory")
    parser.add_argument("--train_file", type=str, default="./data/raw/merlin/train.parquet")
    parser.add_argument("--valid_file", type=str, default="./data/raw/merlin/valid.parquet")
    parser.add_argument("--test_file", type=str, default="./data/raw/merlin/test.parquet")

    parser.add_argument("--user_col", type=str, default="user_session")
    parser.add_argument("--items_col", type=str, default="item_id_list_seq")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["last", "sliding"],
        default="last",
        help="How to expand sequences into training samples",
    )
    parser.add_argument("--max_history_window", type=int, default=10, help="Max history length for sliding strategy")
    parser.add_argument("--max_history_write", type=int, default=20, help="Max history length written to .inter files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = os.path.join(args.output_path, args.dataset)
    check_path(out_dir)

    train_df = pd.read_parquet(args.train_file, engine="pyarrow")
    valid_df = pd.read_parquet(args.valid_file, engine="pyarrow")
    test_df = pd.read_parquet(args.test_file, engine="pyarrow")

    required = [args.user_col, args.items_col]
    _ensure_columns(train_df, args.train_file, required)
    _ensure_columns(valid_df, args.valid_file, required)
    _ensure_columns(test_df, args.test_file, required)

    user2index: Dict[str, int] = {}
    item2index: Dict[str, int] = {}

    user2items_all: Dict[int, List[int]] = {}
    for split_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        split_user2items = _write_split_inter(
            df=df,
            out_path=out_dir,
            dataset=args.dataset,
            split=split_name,
            user2index=user2index,
            item2index=item2index,
            user_col=args.user_col,
            items_col=args.items_col,
            strategy=args.strategy,
            max_history_window=args.max_history_window,
            max_history_write=args.max_history_write,
        )
        for uid, items in split_user2items.items():
            user2items_all.setdefault(uid, []).extend(items)

    # Minimal item features so rq/text2emb/amazon_text2emb.py can run.
    # We only have IDs, so we use the original item ID as "title".
    item2feature: Dict[str, Dict[str, str]] = {}
    for original_item_id, mapped_id in item2index.items():
        item2feature[str(mapped_id)] = {"title": original_item_id, "description": ""}

    write_json_file(user2items_all, os.path.join(out_dir, f"{args.dataset}.inter.json"))
    write_json_file(item2feature, os.path.join(out_dir, f"{args.dataset}.item.json"))
    write_json_file({}, os.path.join(out_dir, f"{args.dataset}.review.json"))

    write_remap_index(user2index, os.path.join(out_dir, f"{args.dataset}.user2id"))
    write_remap_index(item2index, os.path.join(out_dir, f"{args.dataset}.item2id"))

    print("===================================================")
    print(" Merlin Dataset Processing Completed Successfully")
    print("===================================================")
    print(f"Output dir: {out_dir}")
    print(f"Users: {len(user2index)}")
    print(f"Items: {len(item2index)}")
    print(f"Strategy: {args.strategy}")


if __name__ == "__main__":
    main()
