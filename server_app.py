"""simd: ServerApp """

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from simd.model import Net

# Where to store checkpoints (persistent, in your home directory)
CHECKPOINT_DIR = Path(
    os.environ.get("SIMD_CHECKPOINT_DIR", os.path.expanduser("~/simd_checkpoints_wandb"))
)
LATEST_CKPT = CHECKPOINT_DIR / "latest.pt"
META_PATH = CHECKPOINT_DIR / "training_meta.json"

# Create ServerApp
app = ServerApp()


def _load_total_rounds() -> int:
    """Load total rounds completed across all runs from meta.json."""
    if not META_PATH.is_file():
        return 0
    try:
        with META_PATH.open("r") as f:
            data = json.load(f)
        return int(data.get("total_rounds_completed", 0))
    except Exception:
        # If file is corrupted or malformed, start counting from zero again.
        return 0


def _save_meta(total_rounds: int, num_server_rounds_config: int) -> None:
    """Save meta information (total rounds, config) to meta.json."""
    meta = {
        "total_rounds_completed": int(total_rounds),
        "num_server_rounds_config": int(num_server_rounds_config),
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("w") as f:
        json.dump(meta, f)
    print(f"[SERVER] Updated meta.json: {meta}")


def load_initial_arrays() -> ArrayRecord:
    """Create global model and (optionally) load from latest checkpoint."""
    model = Net()  # or your CFDGNN class

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if LATEST_CKPT.is_file():
        print(f"[SERVER] Loading checkpoint from {LATEST_CKPT}")
        state_dict = torch.load(LATEST_CKPT, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("[SERVER] No checkpoint found, starting from scratch")

    return ArrayRecord(model.state_dict())


def make_checkpoint_eval_fn(
    base_rounds: int, num_server_rounds_config: int
):
    """Return an evaluate_fn that saves a checkpoint and updates meta each round."""

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        # Convert ArrayRecord to PyTorch state dict
        state_dict = arrays.to_torch_state_dict()

        # Always update 'latest.pt'
        torch.save(state_dict, LATEST_CKPT)

        # Compute "global" round index across all jobs
        global_round = base_rounds + server_round

        # Keep per-round snapshot with global index
        round_path = CHECKPOINT_DIR / f"model_round_{global_round:04d}.pt"
        torch.save(state_dict, round_path)

        # Update meta.json
        _save_meta(total_rounds=global_round, num_server_rounds_config=num_server_rounds_config)

        print(
            f"[SERVER] Saved checkpoint for global round {global_round} "
            f"(this run round {server_round}) to {round_path}"
        )
        return MetricRecord()

    return evaluate


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read config from pyproject.toml
    # Default to 25 if not present
    num_rounds = int(context.run_config.get("num-server-rounds", 2))
    fraction_train = float(context.run_config.get("fraction-train", 0.5))

    # Load how many rounds we've already completed across previous jobs
    base_rounds = _load_total_rounds()
    print(f"[SERVER] Base rounds already completed before this run: {base_rounds}")
    print(f"[SERVER] This run configured for {num_rounds} rounds")

    # Load global model (with optional checkpoint restore)
    arrays = load_initial_arrays()

    # FedAvg strategy
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_available_nodes=2,
    )

    # Run federated training with checkpointing after every round
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=make_checkpoint_eval_fn(
            base_rounds=base_rounds,
            num_server_rounds_config=num_rounds,
        ),
    )

    # Save final model explicitly
    final_state_dict = result.arrays.to_torch_state_dict()
    final_path = CHECKPOINT_DIR / "final_model.pt"
    torch.save(final_state_dict, final_path)

    # Ensure meta.json at least reflects "base + config" as an upper bound
    total_rounds = base_rounds + num_rounds
    if not META_PATH.is_file():
        _save_meta(total_rounds=total_rounds, num_server_rounds_config=num_rounds)

    print(f"[SERVER] Saved final model to {final_path}")
    print(f"[SERVER] Total rounds completed (at least): {total_rounds}")

