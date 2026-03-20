"""
Data preparation for SMILES canonicalization leakage experiment.

Downloads QM9 dataset and extracts three feature groups:
  A: SMILES-derived features (character bigrams) — NON-invariant under SMILES randomization
  B: Morgan fingerprint bits — INVARIANT (graph-derived)
  C: RDKit molecular descriptors — INVARIANT (graph-derived)

Parallels the NPN circuit experiment's data_prep.py structure.
"""

import json
import os
import sys
import urllib.request
import tarfile
import numpy as np
from collections import Counter
from pathlib import Path

# RDKit imports
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# Suppress RDKit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CONFIG_PATH = PROJECT_DIR / "configs" / "default.json"

QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
QM9_RAW = DATA_DIR / "qm9.csv"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def download_qm9():
    """Download QM9 CSV from MoleculeNet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if QM9_RAW.exists():
        print(f"QM9 already downloaded: {QM9_RAW}")
        return
    print(f"Downloading QM9 from {QM9_URL} ...")
    req = urllib.request.Request(QM9_URL, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=120)
    with open(QM9_RAW, 'wb') as f:
        f.write(resp.read())
    print(f"Downloaded to {QM9_RAW} ({QM9_RAW.stat().st_size / 1e6:.1f} MB)")


def load_qm9_csv():
    """Load QM9 CSV, return SMILES list and property matrix."""
    import csv
    smiles_list = []
    properties = []
    prop_names = None

    with open(QM9_RAW, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # QM9 CSV columns: mol_id, smiles, then properties
        # Find smiles column
        smiles_col = None
        for i, h in enumerate(header):
            if 'smiles' in h.lower():
                smiles_col = i
                break
        if smiles_col is None:
            # Fallback: assume column 1
            smiles_col = 1

        # Property columns: everything after smiles that is numeric
        prop_cols = []
        prop_names_list = []
        for i, h in enumerate(header):
            if i <= smiles_col:
                continue
            prop_cols.append(i)
            prop_names_list.append(h.strip())

        for row in reader:
            smi = row[smiles_col].strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            # Re-canonicalize to ensure consistency
            canon_smi = Chem.MolToSmiles(mol)
            try:
                props = [float(row[c]) for c in prop_cols]
            except (ValueError, IndexError):
                continue
            smiles_list.append(canon_smi)
            properties.append(props)

    properties = np.array(properties, dtype=np.float32)
    print(f"Loaded {len(smiles_list)} molecules, {properties.shape[1]} properties")
    print(f"Property columns: {prop_names_list[:12]}")
    return smiles_list, properties, prop_names_list


def compute_smiles_bigrams(smiles_list, top_n=64):
    """
    Compute character bigram frequency features from SMILES strings.
    These are NON-INVARIANT: different SMILES for the same molecule yield different bigrams.

    Returns: (N, top_n) float32 array, bigram_vocab list
    """
    # First pass: count all bigrams across corpus
    bigram_counter = Counter()
    for smi in smiles_list:
        for i in range(len(smi) - 1):
            bigram_counter[smi[i:i+2]] += 1

    # Select top-N most frequent bigrams as vocabulary
    bigram_vocab = [bg for bg, _ in bigram_counter.most_common(top_n)]
    bg_to_idx = {bg: i for i, bg in enumerate(bigram_vocab)}

    # Second pass: compute per-molecule bigram frequency vectors
    features = np.zeros((len(smiles_list), top_n), dtype=np.float32)
    for j, smi in enumerate(smiles_list):
        total = max(len(smi) - 1, 1)
        for i in range(len(smi) - 1):
            bg = smi[i:i+2]
            if bg in bg_to_idx:
                features[j, bg_to_idx[bg]] += 1
        features[j] /= total  # Normalize to frequencies

    print(f"SMILES bigram features: ({features.shape[0]}, {features.shape[1]})")
    print(f"  Top 10 bigrams: {bigram_vocab[:10]}")
    return features, bigram_vocab


def compute_smiles_bigrams_for_strings(smiles_list, bigram_vocab):
    """Compute bigram features for arbitrary SMILES strings using a fixed vocabulary."""
    bg_to_idx = {bg: i for i, bg in enumerate(bigram_vocab)}
    top_n = len(bigram_vocab)
    features = np.zeros((len(smiles_list), top_n), dtype=np.float32)
    for j, smi in enumerate(smiles_list):
        total = max(len(smi) - 1, 1)
        for i in range(len(smi) - 1):
            bg = smi[i:i+2]
            if bg in bg_to_idx:
                features[j, bg_to_idx[bg]] += 1
        features[j] /= total
    return features


def compute_morgan_fingerprints(smiles_list, n_bits=64, radius=2):
    """
    Compute Morgan (circular) fingerprint bits.
    These are INVARIANT: graph-derived, independent of SMILES string.

    Returns: (N, n_bits) float32 array
    """
    features = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for j, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        features[j] = np.array(fp, dtype=np.float32)
    print(f"Morgan fingerprint features: ({features.shape[0]}, {features.shape[1]})")
    return features


def compute_descriptors(smiles_list):
    """
    Compute RDKit molecular descriptors.
    These are INVARIANT: computed from the molecular graph, not the SMILES string.

    Returns: (N, n_desc) float32 array, descriptor_names list
    """
    desc_funcs = [
        ("MolWt", Descriptors.MolWt),
        ("LogP", Descriptors.MolLogP),
        ("TPSA", Descriptors.TPSA),
        ("NumHDonors", Descriptors.NumHDonors),
        ("NumHAcceptors", Descriptors.NumHAcceptors),
        ("NumRotBonds", Descriptors.NumRotatableBonds),
        ("NumAromaticRings", Descriptors.NumAromaticRings),
        ("NumHeavyAtoms", Descriptors.HeavyAtomCount),
        ("FractionCSP3", Descriptors.FractionCSP3),
        ("NumValenceElectrons", Descriptors.NumValenceElectrons),
        ("NumRadicalElectrons", Descriptors.NumRadicalElectrons),
        ("RingCount", Descriptors.RingCount),
        ("BertzCT", Descriptors.BertzCT),
        ("HallKierAlpha", Descriptors.HallKierAlpha),
        ("LabuteASA", Descriptors.LabuteASA),
    ]

    desc_names = [name for name, _ in desc_funcs]
    features = np.zeros((len(smiles_list), len(desc_funcs)), dtype=np.float32)

    for j, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for k, (name, func) in enumerate(desc_funcs):
            try:
                val = func(mol)
                features[j, k] = float(val) if val is not None else 0.0
            except Exception:
                features[j, k] = 0.0

    # Replace any NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Descriptor features: ({features.shape[0]}, {features.shape[1]})")
    print(f"  Descriptors: {desc_names}")
    return features, desc_names


def generate_random_smiles(smiles, n_random=30, max_attempts_per=50):
    """
    Generate random SMILES for a molecule.
    Returns list of unique random SMILES (may be fewer than n_random for small molecules).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles] * n_random

    random_set = set()
    attempts = 0
    while len(random_set) < n_random and attempts < max_attempts_per * n_random:
        rsmi = Chem.MolToSmiles(mol, doRandom=True)
        random_set.add(rsmi)
        attempts += 1

    result = list(random_set)
    # Pad if not enough unique variants
    while len(result) < n_random:
        result.append(result[np.random.randint(len(result))])
    return result[:n_random]


def main():
    config = load_config()
    seed = config["data"]["random_seed"]
    target_name = config["data"]["target"]
    target_idx = config["data"]["target_index"]
    top_n_bigrams = config["data"]["top_n_bigrams"]
    morgan_bits = config["data"]["morgan_bits"]
    morgan_radius = config["data"]["morgan_radius"]
    np.random.seed(seed)

    # Step 1: Download and load QM9
    download_qm9()
    smiles_list, properties, prop_names = load_qm9_csv()

    # Step 2: Extract target
    targets = properties[:, target_idx]
    print(f"\nTarget: {target_name} (column {target_idx})")
    print(f"  Range: [{targets.min():.4f}, {targets.max():.4f}], Mean: {targets.mean():.4f}, Std: {targets.std():.4f}")

    N = len(smiles_list)

    # Step 3: Compute features
    print("\n--- Feature computation ---")

    # Group A: SMILES bigrams (NON-INVARIANT)
    feat_A, bigram_vocab = compute_smiles_bigrams(smiles_list, top_n=top_n_bigrams)

    # Group B: Morgan fingerprints (INVARIANT)
    feat_B = compute_morgan_fingerprints(smiles_list, n_bits=morgan_bits, radius=morgan_radius)

    # Group C: Molecular descriptors (INVARIANT)
    feat_C, desc_names = compute_descriptors(smiles_list)

    # Concatenate all features
    features_all = np.concatenate([feat_A, feat_B, feat_C], axis=1)
    dim_A = feat_A.shape[1]
    dim_B = feat_B.shape[1]
    dim_C = feat_C.shape[1]
    dim_total = features_all.shape[1]

    print(f"\nFeature dimensions: A(bigrams)={dim_A}, B(morgan)={dim_B}, C(desc)={dim_C}, total={dim_total}")

    # Step 4: Split data (stratified by target quartiles)
    split_ratio = config["data"]["split_ratio"]
    indices = np.arange(N)
    np.random.shuffle(indices)

    n_train = int(N * split_ratio[0])
    n_val = int(N * split_ratio[1])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"\nSplit: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Step 5: Normalize features using training statistics
    train_mean = features_all[train_idx].mean(axis=0)
    train_std = features_all[train_idx].std(axis=0)
    train_std[train_std < 1e-8] = 1.0  # Avoid division by zero
    features_norm = (features_all - train_mean) / train_std

    # Step 6: Save
    output_path = DATA_DIR / "prepared.npz"
    np.savez_compressed(
        output_path,
        # Features
        features=features_norm.astype(np.float32),
        features_raw=features_all.astype(np.float32),
        targets=targets.astype(np.float32),
        # SMILES
        smiles=np.array(smiles_list, dtype=object),
        # Splits
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        # Feature group boundaries
        dim_A=np.array(dim_A),
        dim_B=np.array(dim_B),
        dim_C=np.array(dim_C),
        # Normalization stats
        feature_mean=train_mean.astype(np.float32),
        feature_std=train_std.astype(np.float32),
        # Vocabulary for recomputing bigrams on random SMILES
        bigram_vocab=np.array(bigram_vocab, dtype=object),
        desc_names=np.array(desc_names, dtype=object),
        # Target info
        target_name=np.array(target_name),
    )
    print(f"\nSaved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
