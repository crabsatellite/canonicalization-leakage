"""
Compute information-theoretic and structural measures for all 616,126 NPN5 classes.

Measures:
  1. Shannon entropy of truth table bits
  2. Walsh-Hadamard spectral entropy
  3. Lempel-Ziv complexity (LZ76)
  4. Run-length complexity (number of runs)
  5. gzip compression ratio (batched by circuit size)
  6. Algebraic degree (ANF)
  7. Nonlinearity (distance to nearest affine function)
  8. Autocorrelation absolute sum
  9. Sensitivity (max over inputs of local sensitivity)
  10. Influence (sum of individual variable influences)
"""

import csv
import json
import os
import time
import numpy as np
from collections import Counter


def load_data(csv_path: str):
    truth_tables = []
    circuit_sizes = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tt = int(row["truth_table_hex"], 16)
            ng = int(row["n_gates"])
            truth_tables.append(tt)
            circuit_sizes.append(ng)
    return np.array(truth_tables, dtype=np.uint32), np.array(circuit_sizes, dtype=np.int32)


def tt_to_bits(tt: int, n: int = 32) -> np.ndarray:
    return np.array([(tt >> i) & 1 for i in range(n)], dtype=np.int8)


def batch_tt_to_bits(tts: np.ndarray) -> np.ndarray:
    """Shape: (N, 32)."""
    N = len(tts)
    bits = np.zeros((N, 32), dtype=np.int8)
    for bit in range(32):
        bits[:, bit] = (tts >> bit) & 1
    return bits


# --- Measure 1: Shannon Entropy ---
def shannon_entropy(bits_matrix: np.ndarray) -> np.ndarray:
    """H(bits) — Shannon entropy of truth table bits."""
    p1 = bits_matrix.mean(axis=1)
    p0 = 1.0 - p1
    # Avoid log(0)
    h = np.zeros(len(p1))
    mask = (p0 > 0) & (p1 > 0)
    h[mask] = -p0[mask] * np.log2(p0[mask]) - p1[mask] * np.log2(p1[mask])
    return h


# --- Measure 2: Walsh-Hadamard Spectral Entropy ---
def walsh_hadamard_transform(bits_matrix: np.ndarray) -> np.ndarray:
    """Compute Walsh-Hadamard transform of truth tables.
    Input bits are in {0,1}, converted to {1,-1} as f(x) = (-1)^bit.
    """
    # Convert {0,1} to {1,-1}
    f = 1 - 2 * bits_matrix.astype(np.float64)  # shape (N, 32)

    # WHT via butterfly (5 levels for 2^5 = 32 bits)
    n = 32
    for i in range(5):
        half = 1 << i
        for j in range(0, n, 2 * half):
            for k in range(half):
                u = f[:, j + k].copy()
                v = f[:, j + k + half].copy()
                f[:, j + k] = u + v
                f[:, j + k + half] = u - v
    return f  # shape (N, 32), each row is the WHT spectrum


def spectral_entropy(wht: np.ndarray) -> np.ndarray:
    """Entropy of the normalized Walsh-Hadamard power spectrum."""
    power = wht ** 2
    total = power.sum(axis=1, keepdims=True)
    # Avoid division by zero
    total = np.maximum(total, 1e-15)
    p = power / total
    h = np.zeros(len(p))
    mask = p > 0
    log_p = np.zeros_like(p)
    log_p[mask] = np.log2(p[mask])
    h = -(p * log_p).sum(axis=1)
    return h


# --- Measure 3: Lempel-Ziv Complexity (LZ76) ---
def lz76_complexity(bitstring: str) -> int:
    """Lempel-Ziv complexity (1976) — number of distinct phrases in LZ parsing."""
    n = len(bitstring)
    if n == 0:
        return 0
    c = 1  # complexity counter
    l = 1  # current prefix length
    k = 1  # pointer
    kmax = 1
    while k + l <= n:
        # Check if substring s[k:k+l] appears in s[0:k+l-1]
        if bitstring[k:k+l] in bitstring[:k+l-1]:
            l += 1
        else:
            c += 1
            k += l
            l = 1
    return c


def batch_lz76(bits_matrix: np.ndarray) -> np.ndarray:
    """Compute LZ76 complexity for each row."""
    N = len(bits_matrix)
    result = np.zeros(N, dtype=np.int32)
    for i in range(N):
        s = "".join(str(b) for b in bits_matrix[i])
        result[i] = lz76_complexity(s)
    return result


# --- Measure 4: Run-Length Complexity ---
def batch_run_length(bits_matrix: np.ndarray) -> np.ndarray:
    """Number of runs (consecutive identical bits) in truth table."""
    # A run boundary occurs where bit[i] != bit[i+1]
    diffs = np.diff(bits_matrix, axis=1)
    boundaries = np.count_nonzero(diffs, axis=1)
    return boundaries + 1  # number of runs = boundaries + 1


# --- Measure 5: gzip compression ratio ---
def batch_gzip_ratio(bits_matrix: np.ndarray) -> np.ndarray:
    """Compression ratio of truth table bitstring via gzip."""
    import gzip
    N = len(bits_matrix)
    result = np.zeros(N, dtype=np.float64)
    for i in range(N):
        raw = bytes(bits_matrix[i])
        compressed = gzip.compress(raw, compresslevel=9)
        result[i] = len(compressed) / len(raw)
    return result


# --- Measure 6: Algebraic Degree (ANF) ---
def mobius_transform_batch(bits_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the algebraic normal form (Möbius transform / Reed-Muller transform).
    The ANF coefficient for monomial m is: sum_{x subset of m} f(x) mod 2.
    """
    N, n_bits = bits_matrix.shape
    anf = bits_matrix.copy().astype(np.int8)

    for i in range(5):  # 5 variables for 32-bit truth tables
        step = 1 << i
        for j in range(n_bits):
            if j & step:
                anf[:, j] ^= anf[:, j ^ step]

    return anf


def batch_algebraic_degree(bits_matrix: np.ndarray) -> np.ndarray:
    """Algebraic degree = maximum Hamming weight of monomial with nonzero ANF coefficient."""
    anf = mobius_transform_batch(bits_matrix)
    N = len(anf)
    degrees = np.zeros(N, dtype=np.int32)

    # Precompute Hamming weight of each index 0..31
    hw = np.array([bin(i).count('1') for i in range(32)], dtype=np.int32)

    for i in range(N):
        nonzero_indices = np.where(anf[i] != 0)[0]
        if len(nonzero_indices) > 0:
            degrees[i] = hw[nonzero_indices].max()

    return degrees


# --- Measure 7: Nonlinearity ---
def batch_nonlinearity(wht: np.ndarray) -> np.ndarray:
    """Nonlinearity = (2^{n-1}) - max|W_f(a)|/2 for n=5.
    Equivalently, min Hamming distance to any affine function.
    """
    max_abs_wht = np.abs(wht).max(axis=1)
    return (16 - max_abs_wht / 2).astype(np.float64)


# --- Measure 8: Autocorrelation ---
def batch_autocorrelation_sum(bits_matrix: np.ndarray) -> np.ndarray:
    """Sum of |autocorrelation(f, a)| for all nonzero a.
    Autocorrelation r_f(a) = sum_x (-1)^{f(x) xor f(x xor a)}.
    """
    f = 1 - 2 * bits_matrix.astype(np.float64)  # {1, -1}
    N = len(f)
    auto_sum = np.zeros(N, dtype=np.float64)

    for a in range(1, 32):  # nonzero shifts
        # f(x XOR a): permute columns
        shifted = np.zeros_like(f)
        for x in range(32):
            shifted[:, x] = f[:, x ^ a]
        # r_f(a) = sum of f(x) * f(x XOR a)
        r_a = (f * shifted).sum(axis=1)
        auto_sum += np.abs(r_a)

    return auto_sum


# --- Measure 9: Sensitivity ---
def batch_sensitivity(bits_matrix: np.ndarray) -> np.ndarray:
    """Maximum sensitivity: max over inputs x of the number of neighbors
    where f flips (Hamming distance 1 neighbors in {0,1}^5)."""
    N = len(bits_matrix)
    max_sens = np.zeros(N, dtype=np.int32)

    for x in range(32):
        local_sens = np.zeros(N, dtype=np.int32)
        for bit in range(5):
            neighbor = x ^ (1 << bit)
            # f(x) != f(neighbor)
            flips = bits_matrix[:, x] != bits_matrix[:, neighbor]
            local_sens += flips.astype(np.int32)
        max_sens = np.maximum(max_sens, local_sens)

    return max_sens


# --- Measure 10: Total Influence ---
def batch_influence(bits_matrix: np.ndarray) -> np.ndarray:
    """Total influence = sum over variables of Pr[f(x) != f(x^e_i)]."""
    N = len(bits_matrix)
    total_inf = np.zeros(N, dtype=np.float64)

    for bit in range(5):
        flips = np.zeros(N, dtype=np.int32)
        for x in range(32):
            neighbor = x ^ (1 << bit)
            flips += (bits_matrix[:, x] != bits_matrix[:, neighbor]).astype(np.int32)
        total_inf += flips / 32.0  # Pr over uniform x

    return total_inf


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "data", "npn5_circuit_sizes.csv")
    out_path = os.path.join(script_dir, "..", "data", "measures.csv")

    print("Loading data...")
    tts, sizes = load_data(csv_path)
    N = len(tts)
    print(f"Loaded {N} NPN classes")

    print("Converting to bit matrices...")
    bits = batch_tt_to_bits(tts)

    t0 = time.time()

    print("Computing Walsh-Hadamard transform...")
    wht = walsh_hadamard_transform(bits)

    print("Computing measures...")
    measures = {}

    measures["shannon_entropy"] = shannon_entropy(bits)
    print(f"  shannon_entropy: done ({time.time()-t0:.1f}s)")

    measures["spectral_entropy"] = spectral_entropy(wht)
    print(f"  spectral_entropy: done ({time.time()-t0:.1f}s)")

    measures["lz76_complexity"] = batch_lz76(bits)
    print(f"  lz76_complexity: done ({time.time()-t0:.1f}s)")

    measures["run_length"] = batch_run_length(bits)
    print(f"  run_length: done ({time.time()-t0:.1f}s)")

    measures["gzip_ratio"] = batch_gzip_ratio(bits)
    print(f"  gzip_ratio: done ({time.time()-t0:.1f}s)")

    measures["algebraic_degree"] = batch_algebraic_degree(bits)
    print(f"  algebraic_degree: done ({time.time()-t0:.1f}s)")

    measures["nonlinearity"] = batch_nonlinearity(wht)
    print(f"  nonlinearity: done ({time.time()-t0:.1f}s)")

    measures["autocorrelation_sum"] = batch_autocorrelation_sum(bits)
    print(f"  autocorrelation_sum: done ({time.time()-t0:.1f}s)")

    measures["sensitivity"] = batch_sensitivity(bits)
    print(f"  sensitivity: done ({time.time()-t0:.1f}s)")

    measures["total_influence"] = batch_influence(bits)
    print(f"  total_influence: done ({time.time()-t0:.1f}s)")

    total_time = time.time() - t0
    print(f"\nAll measures computed in {total_time:.1f}s")

    print(f"Writing results to {out_path}...")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["truth_table_hex", "n_gates"] + list(measures.keys())
        writer.writerow(header)
        for i in range(N):
            row = [f"0x{tts[i]:08x}", sizes[i]]
            for m in measures:
                row.append(f"{measures[m][i]:.6f}")
            writer.writerow(row)

    print(f"Done. {N} rows written.")

    print("\n=== Correlation with circuit size ===")
    from scipy.stats import spearmanr, pearsonr
    for name, vals in measures.items():
        r_s, p_s = spearmanr(sizes, vals)
        r_p, p_p = pearsonr(sizes, vals)
        print(f"{name:>25s}:  Spearman r={r_s:+.4f} (p={p_s:.2e})  Pearson r={r_p:+.4f} (p={p_p:.2e})")


if __name__ == "__main__":
    main()
