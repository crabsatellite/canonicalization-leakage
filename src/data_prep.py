"""
Data preparation: load Knuth/Goucher NPN5 data, compute multi-representation
feature vectors, split into train/val/test.

Three representations:
  A) Truth table bits (32) + Fourier spectrum (32) + handcrafted measures (10) = 74-dim
  B) Circuit DAG (for GNN) — separate module
  C) ANF monomial sequence (for Transformer) — separate module

This module handles (A) and the shared train/val/test split.
"""

import os
import csv
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_measures_data(measures_csv: str):
    """Load the measures CSV (truth tables + 10 measures + circuit size)."""
    truth_tables = []
    circuit_sizes = []
    measure_names = None
    measures_list = []

    with open(measures_csv, "r") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        measure_names = [f for f in fields if f not in ("truth_table_hex", "n_gates")]

        for row in reader:
            tt = int(row["truth_table_hex"], 16)
            truth_tables.append(tt)
            circuit_sizes.append(int(float(row["n_gates"])))
            measures_list.append([float(row[m]) for m in measure_names])

    return (np.array(truth_tables, dtype=np.uint32),
            np.array(circuit_sizes, dtype=np.int32),
            np.array(measures_list, dtype=np.float32),
            measure_names)


def tt_to_bits_batch(tts: np.ndarray) -> np.ndarray:
    """Convert truth table integers to bit matrices. Shape: (N, 32)."""
    N = len(tts)
    bits = np.zeros((N, 32), dtype=np.float32)
    for bit in range(32):
        bits[:, bit] = (tts >> bit) & 1
    return bits


def walsh_hadamard_batch(bits: np.ndarray) -> np.ndarray:
    """Compute Walsh-Hadamard transform. Input: {0,1} bits → {+1,-1} → WHT."""
    f = 1.0 - 2.0 * bits  # {0,1} → {+1,-1}
    n = 32
    for i in range(5):
        half = 1 << i
        for j in range(0, n, 2 * half):
            for k in range(half):
                u = f[:, j + k].copy()
                v = f[:, j + k + half].copy()
                f[:, j + k] = u + v
                f[:, j + k + half] = u - v
    return f / 32.0  # normalize


def mobius_transform_batch(bits: np.ndarray) -> np.ndarray:
    """Compute algebraic normal form (ANF) coefficients via Möbius transform."""
    anf = bits.copy().astype(np.float32)
    for i in range(5):
        step = 1 << i
        for j in range(32):
            if j & step:
                anf[:, j] = np.abs(anf[:, j] - anf[:, j ^ step])  # XOR as float
    return anf


def build_features(tts: np.ndarray, measures: np.ndarray) -> np.ndarray:
    """
    Build the 96-dim feature vector:
      [truth_table_bits(32) | fourier_spectrum(32) | anf_coeffs(32) | measures(10)]
    Wait — let's actually use: bits(32) + fourier(32) + measures(10) = 74-dim
    ANF is redundant with bits (invertible transform). Keep it separate for Transformer.
    """
    bits = tt_to_bits_batch(tts)
    fourier = walsh_hadamard_batch(bits.copy())

    # Standardize measures
    m_mean = measures.mean(axis=0)
    m_std = measures.std(axis=0) + 1e-8
    measures_norm = (measures - m_mean) / m_std

    features = np.concatenate([bits, fourier, measures_norm], axis=1)
    return features.astype(np.float32), m_mean, m_std


def split_data(N: int, ratios=(0.7, 0.15, 0.15), seed=42):
    """Return train/val/test index arrays."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    n_train = int(N * ratios[0])
    n_val = int(N * ratios[1])
    return indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]


def prepare_dataloaders(features, targets, train_idx, val_idx, test_idx,
                        batch_size=2048, device="cuda"):
    """Create PyTorch DataLoaders."""
    def make_loader(idx, shuffle):
        X = torch.tensor(features[idx], dtype=torch.float32)
        y = torch.tensor(targets[idx], dtype=torch.float32)
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(device == "cuda"))

    return (make_loader(train_idx, True),
            make_loader(val_idx, False),
            make_loader(test_idx, False))


def main():
    """Prepare and save all data for training."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "configs", "default.json")
    with open(config_path) as f:
        config = json.load(f)

    measures_csv = os.path.join(script_dir, "..", config["data"]["measures_csv"])
    out_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading measures data...")
    tts, sizes, measures, measure_names = load_measures_data(measures_csv)
    N = len(tts)
    print(f"  {N} NPN classes, {len(measure_names)} measures")
    print(f"  Circuit size range: {sizes.min()} - {sizes.max()}")
    print(f"  Distribution: {np.bincount(sizes)}")

    print("Building features...")
    features, m_mean, m_std = build_features(tts, measures)
    print(f"  Feature shape: {features.shape}")

    print("Splitting data...")
    train_idx, val_idx, test_idx = split_data(N, seed=config["data"]["random_seed"])
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Save
    np.savez_compressed(os.path.join(out_dir, "prepared.npz"),
                        features=features,
                        targets=sizes.astype(np.float32),
                        truth_tables=tts,
                        train_idx=train_idx,
                        val_idx=val_idx,
                        test_idx=test_idx,
                        measure_mean=m_mean,
                        measure_std=m_std)

    print(f"  Saved to {out_dir}/prepared.npz")

    # Quick sanity: what's the baseline accuracy if we always predict the mode?
    mode = np.bincount(sizes).argmax()
    mode_acc = (sizes == mode).mean()
    print(f"\n  Mode baseline: always predict {mode} → accuracy {mode_acc:.4f}")
    print(f"  linear baseline: r = {config['linear_baseline_r']}")


if __name__ == "__main__":
    main()
