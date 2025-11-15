"""simd: CFD MeshGraphNet-style GNN using PyTorch Geometric."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader


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


def train(net: Net, trainloader: DataLoader, epochs: int, device: torch.device) -> float:
    """Train the model on the training set, return average loss."""

    net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)

    net.train()
    total_loss = 0.0
    total_graphs = 0

    for _ in range(epochs):
        for batch in trainloader:
            batch = batch.to(device)
            pred = net(batch)  # (sumN, 5)
            loss = loss_fn(pred, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = batch.num_graphs  # number of graphs in this batch
            total_loss += float(loss.item()) * bs
            total_graphs += bs

    avg_trainloss = total_loss / max(1, total_graphs)
    return avg_trainloss


def test(net: Net, testloader: DataLoader, device: torch.device):
    """Evaluate the model on the validation set.

    Returns:
      loss, accuracy (we don't have a real accuracy metric, so 0.0)
    """
    net.to(device)
    loss_fn = nn.MSELoss()

    net.eval()
    total_loss = 0.0
    total_graphs = 0

    with torch.no_grad():
        for batch in testloader:
            batch = batch.to(device)
            pred = net(batch)
            loss = loss_fn(pred, batch.y)
            bs = batch.num_graphs
            total_loss += float(loss.item()) * bs
            total_graphs += bs

    avg_loss = total_loss / max(1, total_graphs)
    accuracy = 0.0  # placeholder, no classification here
    return avg_loss, accuracy
