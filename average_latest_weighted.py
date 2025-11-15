
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

def find_shard_checkpoints(
    ckpt_root: Path,
    shard_prefix: str = "shard",
    filename: str = "latest.pt",
) -> list[Path]:
    """Return list of checkpoint paths like <ckpt_root>/shard*/latest.pt."""
    if not ckpt_root.is_dir():
        raise SystemExit(f"Checkpoint root does not exist or is not a directory: {ckpt_root}")

    shard_dirs = sorted(
        d for d in ckpt_root.iterdir()
        if d.is_dir() and d.name.startswith(shard_prefix)
    )

    ckpts: list[Path] = []
    for d in shard_dirs:
        p = d / filename
        if p.is_file():
            ckpts.append(p)
        else:
            print(f"[WARN] No {filename} in {d}, skipping")

    if not ckpts:
        raise SystemExit(
            f"No '{filename}' files found under {ckpt_root} with prefix '{shard_prefix}'"
        )

    print(f"[INFO] Found {len(ckpts)} shard checkpoints:")
    for p in ckpts:
        print(f"  - {p}")

    return ckpts


def average_checkpoints(ckpt_paths: list[Path]) -> dict:
    """Return averaged state_dict from a list of checkpoint paths."""
    num_shards = len(ckpt_paths)
    print(f"[INFO] Averaging {num_shards} checkpoints")

    # Load first checkpoint to initialize accumulator
    first_sd = torch.load(ckpt_paths[0], map_location="cpu")
    avg_state = {k: v.clone().float() for k, v in first_sd.items()}

    # Zero out accumulator
    for k in avg_state:
        avg_state[k].zero_()

    # Accumulate sum of parameters
    for idx, ckpt_path in enumerate(ckpt_paths):
        print(f"[INFO] Loading shard {idx+1}/{num_shards} from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")
        for k, v in sd.items():
            avg_state[k] += v.float()

    # Divide by number of shards to get mean
    for k in avg_state:
        avg_state[k] /= float(num_shards)

    return avg_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-root",
        type=str,
        default=os.path.expanduser("~/simd_checkpoints_wandb"),
        help="Root directory containing shard subdirs (shard0, shard1, ...).",
    )
    parser.add_argument(
        "--shard-prefix",
        type=str,
        default="shard",
        help="Prefix of shard directories (default: 'shard').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.expanduser("~/simd_checkpoints_wandb/global_avg_latest.pt"),
        help="Path to save the averaged state_dict.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="latest.pt",
        help="Checkpoint filename inside each shard directory (default: latest.pt).",
    )

    args = parser.parse_args()

    ckpt_root = Path(args.ckpt_root).expanduser()
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_paths = find_shard_checkpoints(
        ckpt_root=ckpt_root,
        shard_prefix=args.shard_prefix,
        filename=args.filename,
    )

    avg_state = average_checkpoints(ckpt_paths)

    torch.save(avg_state, out_path)
    print(f"[DONE] Saved averaged model to: {out_path}")


if __name__ == "__main__":
    main()
