"""
Train GIN (Graph Isomorphism Network) on optimal circuit DAGs to predict circuit complexity.

Two pooling variants tested:
  - sum:  GNN embedding = sum of node embeddings (can trivially count nodes → near-perfect)
  - mean: GNN embedding = mean of node embeddings (must learn structural patterns beyond size)

The mean pooling result is the scientifically interesting one: it answers whether
circuit topology encodes useful structural information beyond just the gate count.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from scipy.stats import spearmanr, pearsonr


class GINModel(nn.Module):
    """Graph Isomorphism Network for circuit complexity prediction."""

    def __init__(self, in_dim=6, hidden_dim=64, num_layers=4, dropout=0.1,
                 pool_type="mean"):
        super().__init__()
        self.pool_type = pool_type
        self.num_layers = num_layers

        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_d = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # JumpingKnowledge: concatenate all layer outputs
        jk_dim = hidden_dim * num_layers

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        layer_outputs = []
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = torch.relu(h)
            h = self.dropout(h)
            layer_outputs.append(h)

        # Pool each layer separately, then concatenate (JumpingKnowledge)
        pool_fn = global_add_pool if self.pool_type == "sum" else global_mean_pool
        pooled = [pool_fn(layer_out, batch) for layer_out in layer_outputs]
        graph_emb = torch.cat(pooled, dim=-1)

        return self.head(graph_emb).squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, data.y.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        n += data.num_graphs
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred = []
    all_true = []
    total_loss = 0
    n = 0
    criterion = nn.MSELoss()

    for data in loader:
        data = data.to(device)
        pred = model(data)
        target = data.y.squeeze(-1)
        loss = criterion(pred, target)
        all_pred.append(pred.cpu().numpy())
        all_true.append(target.cpu().numpy())
        total_loss += loss.item() * data.num_graphs
        n += data.num_graphs

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    return {
        "loss": total_loss / n,
        "mse": float(((pred - true) ** 2).mean()),
        "mae": float(np.abs(pred - true).mean()),
        "spearman_r": float(spearmanr(true, pred)[0]),
        "pearson_r": float(pearsonr(true, pred)[0]),
        "exact_match": float((np.round(pred) == true).mean()),
    }


def train_gnn(pool_type="mean", config=None):
    """Train GIN with specified pooling type."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if config is None:
        with open(os.path.join(script_dir, "..", "configs", "default.json")) as f:
            config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load graph data
    graphs_path = os.path.join(script_dir, "..", "data", "circuit_graphs.pt")
    print(f"Loading graphs from {graphs_path}...")
    graphs = torch.load(graphs_path, weights_only=False)
    print(f"  Loaded {len(graphs)} graphs")

    # Load split indices
    prepared = np.load(os.path.join(script_dir, "..", "data", "prepared.npz"))
    train_idx = prepared["train_idx"]
    val_idx = prepared["val_idx"]
    test_idx = prepared["test_idx"]

    gc = config["models"]["gnn"]
    bs = gc["batch_size"]

    # Create DataLoaders
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_graphs, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=bs, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Model
    model = GINModel(
        in_dim=6,
        hidden_dim=gc["hidden_dim"],
        num_layers=gc["num_layers"],
        dropout=gc["dropout"],
        pool_type=pool_type,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"GIN ({pool_type} pool): {param_count:,} params")

    optimizer = optim.AdamW(model.parameters(), lr=gc["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=gc["epochs"])
    criterion = nn.MSELoss()

    # Training
    best_val_r = -1
    best_epoch = 0
    history = []

    t0 = time.time()
    for epoch in range(1, gc["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["spearman_r"] > best_val_r:
            best_val_r = val_metrics["spearman_r"]
            best_epoch = epoch
            ckpt_path = os.path.join(script_dir, "..", "checkpoints",
                                     f"gnn_{pool_type}_best.pt")
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{gc['epochs']} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_r_s={val_metrics['spearman_r']:.4f} | "
                  f"val_mae={val_metrics['mae']:.4f} | "
                  f"val_exact={val_metrics['exact_match']:.4f} | "
                  f"{elapsed:.0f}s")

    # Load best and evaluate on test
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)

    elapsed_total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  GIN ({pool_type} pooling)")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Training time: {elapsed_total:.1f}s")
    print(f"  Test Spearman r:    {test_metrics['spearman_r']:.4f}")
    print(f"  Test Pearson r:     {test_metrics['pearson_r']:.4f}")
    print(f"  Test MAE:           {test_metrics['mae']:.4f}")
    print(f"  Test Exact match:   {test_metrics['exact_match']:.4f}")
    print(f"  linear baseline r:  {config['linear_baseline_r']:.4f}")
    print(f"  MLP best r_s:       0.790 (deep, 468K params)")
    improvement = test_metrics["pearson_r"] - config["linear_baseline_r"]
    print(f"  Improvement over baseline: {improvement:+.4f}")
    print(f"{'=' * 60}")

    # Save results
    results = {
        "mode": f"gnn_{pool_type}",
        "pool_type": pool_type,
        "best_epoch": best_epoch,
        "training_time_s": elapsed_total,
        "params": param_count,
        "test_metrics": test_metrics,
        "linear_baseline_r": config["linear_baseline_r"],
        "history": history,
    }
    results_path = os.path.join(script_dir, "..", "data", f"gnn_{pool_type}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return model, test_metrics


if __name__ == "__main__":
    pool = sys.argv[1] if len(sys.argv) > 1 else "mean"
    print(f"\n{'#' * 60}")
    print(f"# Training GIN ({pool} pooling)")
    print(f"{'#' * 60}\n")
    train_gnn(pool_type=pool)
