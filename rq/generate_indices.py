import argparse
import collections
import json
import os
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# IMPORTANT: this script is intended to be run as `python rq/generate_indices.py`
# so that `datasets.py` and `models/` are imported from the `rq/` directory.
from datasets import EmbDataset
from models.rqvae import RQVAE


def check_collision(all_indices_str: np.ndarray) -> bool:
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str: np.ndarray) -> DefaultDict[str, int]:
    indices_count: DefaultDict[str, int] = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str: np.ndarray) -> List[List[int]]:
    index2id: Dict[str, List[int]] = {}
    for i, index in enumerate(all_indices_str):
        index2id.setdefault(index, []).append(i)

    collision_item_groups = []
    for index, ids in index2id.items():
        if len(ids) > 1:
            collision_item_groups.append(ids)
    return collision_item_groups


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SID indices (.index.json) from an RQ-VAE checkpoint")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained RQ-VAE checkpoint .pth")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name/prefix for output file (default: inferred from args.data_path in checkpoint)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Output directory to write <dataset>.index.json (default: dirname(args.data_path) from checkpoint)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Explicit output file path (default: <root>/<dataset>.index.json)",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda[:N]")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=None, help="Default: from checkpoint args or 0")
    parser.add_argument("--max_collision_passes", type=int, default=20, help="Max SK refinement passes")
    parser.add_argument(
        "--sk_epsilon",
        type=float,
        default=None,
        help="Override final-layer sk_epsilon when resolving collisions (default: keep ckpt, but min 0.003)",
    )
    return parser.parse_args()


def infer_dataset_from_path(data_path: str) -> str:
    base = os.path.basename(data_path)
    if base.endswith(".npy"):
        base = base[: -len(".npy")]
    return base or "dataset"


def load_checkpoint(ckpt_path: str) -> Tuple[Any, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint format at {ckpt_path}")
    if "args" not in ckpt or "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint at {ckpt_path} must contain 'args' and 'state_dict'")
    return ckpt["args"], ckpt["state_dict"]


def build_model(ckpt_args: Any, embedding_dim: int) -> RQVAE:
    return RQVAE(
        in_dim=embedding_dim,
        num_emb_list=ckpt_args.num_emb_list,
        e_dim=ckpt_args.e_dim,
        layers=ckpt_args.layers,
        dropout_prob=ckpt_args.dropout_prob,
        bn=ckpt_args.bn,
        loss_type=ckpt_args.loss_type,
        quant_loss_weight=ckpt_args.quant_loss_weight,
        kmeans_init=ckpt_args.kmeans_init,
        kmeans_iters=ckpt_args.kmeans_iters,
        sk_epsilons=ckpt_args.sk_epsilons,
        sk_iters=ckpt_args.sk_iters,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    ckpt_args, state_dict = load_checkpoint(args.ckpt_path)
    data_path = getattr(ckpt_args, "data_path", None)
    if not data_path:
        raise ValueError("Checkpoint args missing required 'data_path'")

    dataset_name = args.dataset or infer_dataset_from_path(str(data_path))
    out_root = args.root or os.path.dirname(str(data_path))
    out_file = args.output_file or os.path.join(out_root, f"{dataset_name}.index.json")
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    dataset = EmbDataset(str(data_path))

    model = build_model(ckpt_args, dataset.dim)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(getattr(ckpt_args, "num_workers", 0) or 0)

    data_loader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    all_indices: List[List[str]] = []
    all_indices_str: List[str] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Encoding"):
            batch = batch.to(device)
            indices = model.get_indices(batch, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices.append(code)
                all_indices_str.append(str(code))

    all_indices_arr = np.array(all_indices, dtype=object)
    all_indices_str_arr = np.array(all_indices_str, dtype=str)

    # Match the original collision-resolution logic: disable SK on all but last layer.
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0

    if args.sk_epsilon is not None:
        model.rq.vq_layers[-1].sk_epsilon = float(args.sk_epsilon)
    elif model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    passes = 0
    while passes < args.max_collision_passes and not check_collision(all_indices_str_arr):
        collision_item_groups = get_collision_item(all_indices_str_arr)
        for collision_items in collision_item_groups:
            d = dataset[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices_arr[item] = code
                all_indices_str_arr[item] = str(code)
        passes += 1

    print("All indices number:", len(all_indices_str_arr))
    print("Max number of conflicts:", max(get_indices_count(all_indices_str_arr).values()))
    tot_item = len(all_indices_str_arr)
    tot_indice = len(set(all_indices_str_arr.tolist()))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)
    print("Collision passes:", passes)

    all_indices_dict: Dict[int, Sequence[str]] = {item: list(indices) for item, indices in enumerate(all_indices_arr.tolist())}
    with open(out_file, "w") as fp:
        json.dump(all_indices_dict, fp)

    print(f"Wrote: {out_file}")


if __name__ == "__main__":
    main()

