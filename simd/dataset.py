"""simd: CFD dataset on JSON + PyTorch Geometric, with job-level sharding."""

from __future__ import annotations

import json
import math
import os
import pathlib
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader

DTYPE = torch.float32


class CfdGraphDataset(Dataset):
    """
    CFD graph dataset backed by step_*.json files.

    JSON structure per step_xxxxx.json:
      node_input   : (N, 8)  [x,y,z,T,p,u_x,u_y,u_z]
      edge_index   : (2, E)  int64
      edge_attr    : (E, 4)  [dx,dy,dz,r] (or similar)
      target_delta : (N, 5)  [ΔT, Δp, Δu_x, Δu_y, Δu_z]

    Normalization stats:
      node_stats.json, edge_stats.json, target_stats.json

    Two levels of partitioning:

      1) Job-level sharding (across Slurm jobs):
         - SIMD_JOB_SHARD_ID     (default 0)
         - SIMD_JOB_NUM_SHARDS   (default 1)

         Each job sees a *contiguous* block of step_*.json.

      2) Client-level partitioning (within a Flower federation):
         - client_id   (0..num_clients-1)
         - num_clients (from Flower, typically num-supernodes)
    """

    def __init__(
        self,
        root_dir: str | pathlib.Path,
        client_id: int,
        num_clients: int,
        split: str = "train",
        train_ratio: float = 0.8,
    ) -> None:
        super().__init__()
        self.root_dir = pathlib.Path(root_dir)
        self.client_id = int(client_id)
        self.num_clients = int(num_clients)

        # -----------------------------
        # 1) Collect all step_*.json
        # -----------------------------
        all_files = sorted(self.root_dir.glob("step_*.json"))
        if not all_files:
            raise RuntimeError(f"No step_*.json files found in {self.root_dir}")

        n_all = len(all_files)

        # -----------------------------
        # 2) Job-level sharding
        # -----------------------------
        shard_id = int(os.environ.get("SIMD_JOB_SHARD_ID", "0"))
        num_shards = int(os.environ.get("SIMD_JOB_NUM_SHARDS", "1"))

        if num_shards <= 0:
            raise ValueError(f"SIMD_JOB_NUM_SHARDS must be >=1, got {num_shards}")
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(
                f"SIMD_JOB_SHARD_ID must be in [0, {num_shards - 1}], got {shard_id}"
            )

        files_per_shard = math.ceil(n_all / num_shards)
        start_idx = shard_id * files_per_shard
        end_idx = min(n_all, start_idx + files_per_shard)
        shard_files = all_files[start_idx:end_idx]

        if not shard_files:
            raise RuntimeError(
                f"No files assigned to shard_id={shard_id} with num_shards={num_shards}"
            )

        # -----------------------------
        # 3) Client-level partitioning
        # -----------------------------
        client_files = shard_files[self.client_id :: self.num_clients]
        if not client_files:
            raise RuntimeError(
                f"No files assigned to client_id={self.client_id} "
                f"with num_clients={self.num_clients} for shard {shard_id}/{num_shards}"
            )

        # -----------------------------
        # 4) Train/val split
        # -----------------------------
        n = len(client_files)
        n_train = max(1, int(train_ratio * n))

        if split == "train":
            self.files = client_files[:n_train]
        else:
            self.files = client_files[n_train:]
            # Edge-case: if val would be empty, keep at least one
            if not self.files:
                self.files = client_files[-max(1, n // 5) :]

        # -----------------------------
        # 5) Load normalization stats
        # -----------------------------
        def _load_stats(name: str):
            p = self.root_dir / name
            if not p.is_file():
                raise RuntimeError(f"Missing stats file: {p}")
            with open(p, "r") as f:
                d = json.load(f)
            mu = torch.tensor(d["mu"], dtype=DTYPE)
            sd = torch.tensor(d["std"], dtype=DTYPE)
            return mu, sd

        self.node_mu, self.node_sd = _load_stats("node_stats.json")
        self.edge_mu, self.edge_sd = _load_stats("edge_stats.json")
        self.tgt_mu, self.tgt_sd = _load_stats("target_stats.json")

        assert self.node_mu.numel() == 8 and self.node_sd.numel() == 8
        assert self.edge_mu.numel() == 4 and self.edge_sd.numel() == 4
        assert self.tgt_mu.numel() == 5 and self.tgt_sd.numel() == 5

        print(
            f"[DATA] shard_id={shard_id}/{num_shards}, "
            f"client_id={self.client_id}/{self.num_clients}, "
            f"split={split}, n_files={len(self.files)}"
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> PyGData:
        p = self.files[idx]
        with open(p, "r") as f:
            s = json.load(f)

        node = torch.from_numpy(
            np.asarray(s["node_input"], dtype=np.float32)
        )  # (N, 8)
        edge_index = torch.from_numpy(
            np.asarray(s["edge_index"], dtype=np.int64)
        )  # (2, E)
        edge_attr = torch.from_numpy(
            np.asarray(s["edge_attr"], dtype=np.float32)
        )  # (E, 4)
        target = torch.from_numpy(
            np.asarray(s["target_delta"], dtype=np.float32)
        )  # (N, 5)

        # standardize
        node = (node - self.node_mu) / self.node_sd
        edge_attr = (edge_attr - self.edge_mu) / self.edge_sd
        target = (target - self.tgt_mu) / self.tgt_sd

        data = PyGData(
            x=node,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=target,
        )
        return data


def load_data(partition_id: int, num_partitions: int) -> Tuple[DataLoader, DataLoader]:
    """Load CFD graph data for one Flower client.

    partition_id   : this client's ID (0..num_partitions-1)
    num_partitions : total number of federated clients (per job)
    """

    root_dir = pathlib.Path(
        os.environ.get(
            "CFD_JSON_ROOT",
            os.path.expanduser("~/cfd-metadata-json"),
        )
    )

    train_ds = CfdGraphDataset(
        root_dir=root_dir,
        client_id=partition_id,
        num_clients=num_partitions,
        split="train",
    )
    val_ds = CfdGraphDataset(
        root_dir=root_dir,
        client_id=partition_id,
        num_clients=num_partitions,
        split="val",
    )

    trainloader = DataLoader(train_ds, batch_size=1, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    return trainloader, valloader
