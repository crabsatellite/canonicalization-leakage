"""
Matched-volume control experiment for SMILES canonicalization leakage.

Three conditions at 2x volume (paralleling Section 4.5):
  1. SMILES augmentation: canonical + 1 random SMILES (group-consistent)
  2. Data duplication: canonical repeated 2x (same volume, no new info)
  3. Noise augmentation: canonical + copy with Gaussian noise on features

If invariance gain is from group-consistent transformation (not mere volume),
only SMILES augmentation should recover invariant performance.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
from pathlib import Path

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from data_prep import compute_smiles_bigrams_for_strings, generate_random_smiles
from train_mlp import MolMLP, load_config, evaluate
from smiles_invariance_test import run_invariance_test

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CKPT_DIR = PROJECT_DIR / "checkpoints"


def prepare_duplicated_data(features, targets):
    """Simply duplicate the canonical data 2x."""
    return np.concatenate([features, features], axis=0), np.concatenate([targets, targets])


def prepare_noise_data(features, targets, sigma=0.1):
    """Canonical + noisy copy."""
    noisy = features + np.random.randn(*features.shape).astype(np.float32) * sigma
    return np.concatenate([features, noisy], axis=0), np.concatenate([targets, targets])


def prepare_smiles_augmented_data(features, targets, smiles, dim_A,
                                   bigram_vocab, feature_mean, feature_std):
    """Canonical + 1 random SMILES copy (group-consistent)."""
    N = len(features)
    aug_features = features.copy()

    print("Generating SMILES-augmented copies...")
    for j in range(N):
        smi = str(smiles[j])
        rand_smiles = generate_random_smiles(smi, n_random=1)
        bigram_feats = compute_smiles_bigrams_for_strings(rand_smiles, bigram_vocab)
        bigram_feats_norm = (bigram_feats[0] - feature_mean[:dim_A]) / feature_std[:dim_A]
        aug_features[j, :dim_A] = bigram_feats_norm

    return (np.concatenate([features, aug_features], axis=0),
            np.concatenate([targets, targets]))


def train_control(X_train, y_train, X_val, y_val, X_test, y_test, tag, config):
    """Train a model on provided data and evaluate."""
    mc = config["models"]["mlp"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    model = MolMLP(input_dim, mc["hidden_dims"], mc["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=mc["lr"],
                                  weight_decay=mc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, mc["epochs"])
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                       torch.tensor(y_train, dtype=torch.float32)),
        batch_size=mc["batch_size"], shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                       torch.tensor(y_val, dtype=torch.float32)),
        batch_size=mc["batch_size"])
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                       torch.tensor(y_test, dtype=torch.float32)),
        batch_size=mc["batch_size"])

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{tag}_best.pt"
    best_val_rs = -1

    for epoch in range(mc["epochs"]):
        model.train()
        total_loss = 0
        n = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        if val_metrics["spearman_r"] > best_val_rs:
            best_val_rs = val_metrics["spearman_r"]
            torch.save(model.state_dict(), ckpt_path)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | val_rs={val_metrics['spearman_r']:.4f}")

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    return test_metrics


def main():
    config = load_config()
    np.random.seed(42)
    torch.manual_seed(42)

    data = np.load(DATA_DIR / "prepared.npz", allow_pickle=True)
    features = data["features"]
    targets = data["targets"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    smiles = data["smiles"]
    dim_A = int(data["dim_A"])
    bigram_vocab = list(data["bigram_vocab"])
    feature_mean = data["feature_mean"]
    feature_std = data["feature_std"]

    X_val = features[val_idx]
    y_val = targets[val_idx]
    X_test = features[test_idx]
    y_test = targets[test_idx]

    results = {}

    # Condition 1: SMILES augmentation (group-consistent)
    print(f"\n{'='*60}")
    print("Condition 1: SMILES augmentation (group-consistent)")
    print(f"{'='*60}")
    X_aug, y_aug = prepare_smiles_augmented_data(
        features[train_idx], targets[train_idx], smiles[train_idx],
        dim_A, bigram_vocab, feature_mean, feature_std)
    metrics = train_control(X_aug, y_aug, X_val, y_val, X_test, y_test,
                            "mlp_control_smiles_aug", config)
    results["smiles_augmentation"] = {k: float(v) for k, v in metrics.items()}
    results["smiles_augmentation"]["train_size"] = len(y_aug)
    print(f"Test r_s = {metrics['spearman_r']:.4f}, MAE = {metrics['mae']:.4f}")

    # Condition 2: Data duplication
    print(f"\n{'='*60}")
    print("Condition 2: Data duplication (no new information)")
    print(f"{'='*60}")
    X_dup, y_dup = prepare_duplicated_data(features[train_idx], targets[train_idx])
    metrics = train_control(X_dup, y_dup, X_val, y_val, X_test, y_test,
                            "mlp_control_duplicate", config)
    results["data_duplication"] = {k: float(v) for k, v in metrics.items()}
    results["data_duplication"]["train_size"] = len(y_dup)
    print(f"Test r_s = {metrics['spearman_r']:.4f}, MAE = {metrics['mae']:.4f}")

    # Condition 3: Noise augmentation
    print(f"\n{'='*60}")
    print("Condition 3: Noise augmentation (random perturbation)")
    print(f"{'='*60}")
    X_noise, y_noise = prepare_noise_data(features[train_idx], targets[train_idx], sigma=0.1)
    metrics = train_control(X_noise, y_noise, X_val, y_val, X_test, y_test,
                            "mlp_control_noise", config)
    results["noise_augmentation"] = {k: float(v) for k, v in metrics.items()}
    results["noise_augmentation"]["train_size"] = len(y_noise)
    print(f"Test r_s = {metrics['spearman_r']:.4f}, MAE = {metrics['mae']:.4f}")

    # Run invariance tests on each control model
    print(f"\n{'='*60}")
    print("Running invariance tests on control models...")
    print(f"{'='*60}")
    for tag_name, model_tag in [
        ("smiles_augmentation", "mlp_control_smiles_aug"),
        ("data_duplication", "mlp_control_duplicate"),
        ("noise_augmentation", "mlp_control_noise"),
    ]:
        inv_results = run_invariance_test(model_tag=model_tag, n_test=500, K=30)
        results[tag_name]["smiles_avg_rs"] = inv_results["smiles_averaged_rs"]
        results[tag_name]["mean_std"] = inv_results["mean_prediction_std"]
        results[tag_name]["consistency"] = inv_results["consistency_2dp"]

    # Save
    output_path = DATA_DIR / "augmentation_control.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("MATCHED-VOLUME CONTROL SUMMARY (all at 2x volume)")
    print(f"{'='*60}")
    print(f"{'Condition':<25} {'Val r_s':>8} {'Avg r_s':>8} {'Std':>8} {'Consist':>8}")
    print(f"{'-'*57}")
    for name in ["smiles_augmentation", "data_duplication", "noise_augmentation"]:
        r = results[name]
        print(f"{name:<25} {r['spearman_r']:>8.4f} {r.get('smiles_avg_rs', 'N/A'):>8} "
              f"{r.get('mean_std', 'N/A'):>8} {r.get('consistency', 'N/A'):>8}")


if __name__ == "__main__":
    main()
