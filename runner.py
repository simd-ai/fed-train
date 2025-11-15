#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Any, Dict

HOME = pathlib.Path.home()
STATE_PATH = HOME / "simd_shard_state.json"


def load_state(default_num_shards: int) -> Dict[str, Any]:
    if STATE_PATH.is_file():
        with STATE_PATH.open("r") as f:
            try:
                data = json.load(f)
            except Exception:
                data = {}
    else:
        data = {}
    if "num_shards" not in data:
        data["num_shards"] = int(default_num_shards)
    if "next_shard" not in data:
        data["next_shard"] = 0
    return data


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(state, f)
    tmp.replace(STATE_PATH)


def main() -> None:
    # How many shards in total? You can override per job via env.
    num_shards = int(os.environ.get("SIMD_NUM_SHARDS", "10"))

    state = load_state(num_shards)
    if state["num_shards"] != num_shards:
        print(
            f"[SHARD] WARNING: state.num_shards={state['num_shards']} "
            f"!= requested {num_shards}. Using {state['num_shards']}."
        )
        num_shards = int(state["num_shards"])

    shard_id = int(state["next_shard"])
    next_shard = (shard_id + 1) % num_shards
    state["next_shard"] = next_shard
    save_state(state)

    print(
        f"[SHARD] This job will use shard_id={shard_id} "
        f"out of num_shards={num_shards}"
    )

    # Export env vars for this process and its children
    os.environ["SIMD_JOB_SHARD_ID"] = str(shard_id)
    os.environ["SIMD_JOB_NUM_SHARDS"] = str(num_shards)

    # Set per-shard checkpoint dir
    ckpt_base = os.environ.get(
        "SIMD_CHECKPOINT_BASE", str(HOME / "simd_checkpoints")
    )
    ckpt_dir = pathlib.Path(ckpt_base) / f"shard{shard_id}"
    os.environ["SIMD_CHECKPOINT_DIR"] = str(ckpt_dir)

    print(f"[SHARD] Using checkpoint dir: {ckpt_dir}")

    # Finally run Flower on this shard
    # (flwr CLI is available in the hackathon-venv)
    subprocess.run(["flwr", "run", ".", "cluster-gpu"], check=True)


if __name__ == "__main__":
    main()
