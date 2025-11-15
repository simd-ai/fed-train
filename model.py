"""simd: CFD MeshGraphNet-style GNN using PyTorch Geometric."""

from __future__ import annotations

import os
import socket
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader

# --- Optional Weights & Biases ----------------------------------------------

try:
    import wandb  # type: ignore[import]
except Exception:  # wandb not available or import error
    wandb = None  # type: ignore[assignment]


def _maybe_init_wandb(job_type: str = "train") -> None:
    """Initialize wandb once per process, if available and not disabled.

    Uses environment variables to tag shard and experiment:
      - WANDB_PROJECT         (default: "simd-cfd")
      - WANDB_ENTITY          (optional: your team/user on wandb)
      - WANDB_RUN_NAME        (optional: custom run name)
      - WANDB_RUN_GROUP       (optional: custom group)
      - SIMD_JOB_SHARD_ID     (used as group label if present)
      - SIMD_JOB_NUM_SHARDS   (for info only)
      - WANDB_DISABLED=true   to turn logging off
    """

    if wandb is None:
        return

    if os.environ.get("WANDB_DISABLED", "false").lower() == "true":
        return

    # If a run already exists in this process, reuse it
    if wandb.run is not None:
        return

    project = os.environ.get("WANDB_PROJECT", "simd-cfd")
    entity = os.environ.get("WANDB_ENTITY")  # optional

    shard_id = os.environ.get("SIMD_JOB_SHARD_ID", "0")
    num_shards = os.environ.get("SIMD_JOB_NUM_SHARDS", "1")

    default_group = f"shard{shard_id}"
    default_name = f"shard{shard_id}-{socket.gethostname()}"

    run_name = os.environ.get("WANDB_RUN_NAME", default_name)
    run_group = os.environ.get("WANDB_RUN_GROUP", default_group)

    config = {
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "job_type": job_type,
    }

    if entity:
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=run_group,
            job_type=job_type,
            config=config,
            reinit=True,
        )
    else:
        wandb.init(
            project=project,
            name=run_name,
            group=run_group,
            job_type=job_type,
            config=config,
            reinit=True,
        )


# --- Model -------------------------------------------------------------------


class Net(nn.Module):
    """CFD GNN: MeshGraphNet-style NNConv network.

    Input:
      x          : (N, 8)   node features
      edge_index : (2, E)
      edge_attr  : (E, 4)

    Output:
      y_pred     : (N, 5)   normalized deltas [ΔT, Δp, Δu_x, Δu_y, Δu_z]
    """

    def __init__(
        self,
        node_in: int = 8,
        edge_in: int = 4,
        hidden: int = 128,
        out_dim: int = 5,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        from torch_geometric.nn import NNConv

        # Edge-network for NNConv: edge_attr -> weight matrix
        self.edge_net1 = nn.Sequential(
            nn.Linear(edge_in, hidden * node_in),
            nn.ReLU(),
            nn.Linear(hidden * node_in, hidden * node_in),
        )
        self.conv1 = NNConv(
            in_channels=node_in,
            out_channels=hidden,
            nn=self.edge_net1,
            aggr="mean",
        )

        self.edge_net2 = nn.Sequential(
            nn.Linear(edge_in, hidden * hidden),
            nn.ReLU(),
            nn.Linear(hidden * hidden, hidden * hidden),
        )
        self.conv2 = NNConv(
            in_channels=hidden,
            out_channels=hidden,
            nn=self.edge_net2,
            aggr="mean",
        )

        # Optional third layer
        self.edge_net3 = nn.Sequential(
            nn.Linear(edge_in, hidden * hidden),
            nn.ReLU(),
            nn.Linear(hidden * hidden, hidden * hidden),
        )
        self.conv3 = NNConv(
            in_channels=hidden,
            out_channels=hidden,
            nn=self.edge_net3,
            aggr="mean",
        )

        self.lin_out = nn.Linear(hidden, out_dim)
        self.act = nn.ReLU()
        self.num_layers = num_layers

    def forward(self, data: PyGData) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.act(self.conv1(x, edge_index, edge_attr))
        x = self.act(self.conv2(x, edge_index, edge_attr))
        if self.num_layers >= 3:
            x = self.act(self.conv3(x, edge_index, edge_attr))

        out = self.lin_out(x)  # (N, 5)
        return out


# --- Training / Evaluation ---------------------------------------------------


def _compute_channel_mse_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """Return per-channel sum of squared error and count of nodes.

    pred, target: shape (N, 5)
    Returns:
      (se_per_channel, count_nodes)
    """
    diff = pred - target  # (N, 5)
    # Sum of squared errors per channel
    se = (diff ** 2).sum(dim=0).double().cpu()  # (5,)
    count = diff.shape[0]  # number of nodes N
    return se, count


def train(net: Net, trainloader: DataLoader, epochs: int, device: torch.device) -> float:
    """Train the model on the training set, return average loss.

    Also logs:
      - train/loss
      - train/mse_dT, train/mse_dp, train/mse_dux, train/mse_duy, train/mse_duz
    to Weights & Biases if available.
    """

    net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)

    _maybe_init_wandb(job_type="train")

    net.train()

    # For returning overall average across all epochs and graphs
    global_loss_sum = 0.0
    global_graphs = 0

    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        epoch_graphs = 0

        # per-channel accumulators
        channel_se_sum = torch.zeros(5, dtype=torch.float64)
        channel_node_count = 0

        for batch in trainloader:
            batch = batch.to(device)
            pred = net(batch)  # (sumN, 5)
            loss = loss_fn(pred, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = batch.num_graphs  # number of graphs in this batch
            epoch_loss_sum += float(loss.item()) * bs
            epoch_graphs += bs

            se, count_nodes = _compute_channel_mse_sums(pred, batch.y)
            channel_se_sum += se
            channel_node_count += count_nodes

        # Average loss over graphs for this epoch
        epoch_avg_loss = epoch_loss_sum / max(1, epoch_graphs)
        global_loss_sum += epoch_avg_loss * epoch_graphs
        global_graphs += epoch_graphs

        # Per-channel MSE
        if channel_node_count > 0:
            channel_mse = channel_se_sum / channel_node_count
            dT_mse = float(channel_mse[0].item())
            dp_mse = float(channel_mse[1].item())
            dux_mse = float(channel_mse[2].item())
            duy_mse = float(channel_mse[3].item())
            duz_mse = float(channel_mse[4].item())
        else:
            dT_mse = dp_mse = dux_mse = duy_mse = duz_mse = 0.0

        if wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "train/loss": epoch_avg_loss,
                    "train/mse_dT": dT_mse,
                    "train/mse_dp": dp_mse,
                    "train/mse_dux": dux_mse,
                    "train/mse_duy": duy_mse,
                    "train/mse_duz": duz_mse,
                }
            )

    avg_trainloss = global_loss_sum / max(1, global_graphs)
    return avg_trainloss


def test(net: Net, testloader: DataLoader, device: torch.device):
    """Evaluate the model on the validation set.

    Returns:
      loss, accuracy (we don't have a real accuracy metric, so 0.0)

    Also logs:
      - val/loss
      - val/mse_dT, val/mse_dp, val/mse_dux, val/mse_duy, val/mse_duz
    """

    net.to(device)
    loss_fn = nn.MSELoss()

    _maybe_init_wandb(job_type="eval")

    net.eval()
    total_loss = 0.0
    total_graphs = 0

    channel_se_sum = torch.zeros(5, dtype=torch.float64)
    channel_node_count = 0

    with torch.no_grad():
        for batch in testloader:
            batch = batch.to(device)
            pred = net(batch)
            loss = loss_fn(pred, batch.y)
            bs = batch.num_graphs
            total_loss += float(loss.item()) * bs
            total_graphs += bs

            se, count_nodes = _compute_channel_mse_sums(pred, batch.y)
            channel_se_sum += se
            channel_node_count += count_nodes

    avg_loss = total_loss / max(1, total_graphs)

    if channel_node_count > 0:
        channel_mse = channel_se_sum / channel_node_count
        dT_mse = float(channel_mse[0].item())
        dp_mse = float(channel_mse[1].item())
        dux_mse = float(channel_mse[2].item())
        duy_mse = float(channel_mse[3].item())
        duz_mse = float(channel_mse[4].item())
    else:
        dT_mse = dp_mse = dux_mse = duy_mse = duz_mse = 0.0

    if wandb is not None and wandb.run is not None:
        wandb.log(
            {
                "val/loss": avg_loss,
                "val/mse_dT": dT_mse,
                "val/mse_dp": dp_mse,
                "val/mse_dux": dux_mse,
                "val/mse_duy": duy_mse,
                "val/mse_duz": duz_mse,
            }
        )

    accuracy = 0.0  # placeholder, no classification here
    return avg_loss, accuracy
