"""
Prepare PyG graph dataset from optimal circuit DAGs (knuthies.dat).

Each circuit becomes a directed graph:
  - Nodes: constant-0, 5 input variables, up to 12 gate nodes
  - Node features: one-hot type (6 types: const/input/AND/OR/XOR/ANDNOT)
  - Edges: directed from gate inputs to gate (+ reverse for bidirectional message passing)
  - Target: circuit size (number of gates)

Design note: The GNN sees the optimal circuit structure — an "oracle" representation
that encodes the answer in its node count. We compare sum pooling (trivially encodes
node count) vs. mean pooling (must learn structural patterns beyond size). The mean
pooling result answers: "Does circuit topology encode information beyond gate count?"
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path


# 5-input truth tables for x1..x5
INPUT_TTS = [
    0x0000ffff,  # x1 (bit 4)
    0x00ff00ff,  # x2 (bit 3)
    0x0f0f0f0f,  # x3 (bit 2)
    0x33333333,  # x4 (bit 1)
    0x55555555,  # x5 (bit 0)
]

OP_NAMES = {0: "AND", 1: "OR", 2: "XOR", 3: "ANDNOT"}


def apply_op(opa, op_code, opb):
    ops = {
        0: lambda a, b: a & b,
        1: lambda a, b: a | b,
        2: lambda a, b: a ^ b,
        3: lambda a, b: a & (~b & 0xFFFFFFFF),
    }
    return ops[op_code](opa, opb) & 0xFFFFFFFF


def decode_circuit_graph(data_15bytes):
    """
    Decode 15-byte knuthies.dat entry into a PyG Data object.

    Returns (truth_table, n_gates, Data) or None if degenerate.
    Node types: 0=constant, 1=input, 2=AND, 3=OR, 4=XOR, 5=ANDNOT
    """
    gates = []
    for group in range(3):
        offset = group * 5
        val = 0
        for i in range(5):
            val |= data_15bytes[offset + i] << (8 * i)
        for i in range(4):
            g = (val >> (10 * i)) & 0x3FF
            gates.append(g)

    # Build graph nodes and edges
    # Nodes 0-5: constant-0 + 5 inputs
    node_types = [0] + [1] * 5
    edges_src = []
    edges_dst = []

    calcs = [0x00000000] + INPUT_TTS.copy()
    n_gates = 0

    for i, g in enumerate(gates):
        if g == 0:
            n_gates = i
            break
        op_code = (g >> 8) & 0x3
        i1 = (g & 0xF) + 1
        i2 = ((g >> 4) & 0xF) + 1

        if i1 >= len(calcs) or i2 >= len(calcs):
            return None

        gate_node_idx = 6 + i
        node_types.append(op_code + 2)  # AND=2, OR=3, XOR=4, ANDNOT=5

        # Directed edges: inputs -> gate
        edges_src.extend([i1, i2])
        edges_dst.extend([gate_node_idx, gate_node_idx])

        result = apply_op(calcs[i1], op_code, calcs[i2])
        calcs.append(result)
    else:
        n_gates = 12

    truth_table = calcs[-1] if len(calcs) > 6 else calcs[0]

    num_nodes = 6 + n_gates

    # Node features: one-hot type (6 dim)
    x = torch.zeros(num_nodes, 6)
    for i, t in enumerate(node_types[:num_nodes]):
        x[i, t] = 1.0

    # Make edges bidirectional for message passing
    if len(edges_src) > 0:
        all_src = edges_src + edges_dst
        all_dst = edges_dst + edges_src
        edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    y = torch.tensor([n_gates], dtype=torch.float)

    return (truth_table, n_gates, Data(x=x, edge_index=edge_index, y=y))


def build_trivial_graphs():
    """Build graphs for the 2 trivial NPN classes (0 gates)."""
    graphs = []

    # Constant 0: just the 6 base nodes, no edges
    x = torch.zeros(6, 6)
    x[0, 0] = 1.0  # constant
    for i in range(5):
        x[i + 1, 1] = 1.0  # input
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    y = torch.tensor([0], dtype=torch.float)
    graphs.append((0x00000000, 0, Data(x=x, edge_index=edge_index, y=y)))

    # Identity x1: same structure (0 gates, output = x1)
    x2 = x.clone()
    graphs.append((0x0000ffff, 0, Data(x=x2, edge_index=edge_index.clone(), y=y.clone())))

    return graphs


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dat_path = os.path.join(script_dir, "..", "..", "data", "optimal5", "knuthies.dat")
    out_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    # Load binary data
    raw = Path(dat_path).read_bytes()
    n_entries = len(raw) // 15
    print(f"knuthies.dat: {len(raw)} bytes, {n_entries} entries")

    # Decode all circuits
    results = []
    skipped = 0
    for i in range(n_entries):
        entry = raw[i * 15 : (i + 1) * 15]
        decoded = decode_circuit_graph(entry)
        if decoded is not None:
            results.append(decoded)
        else:
            skipped += 1

    # Add trivial classes
    tt_set = {r[0] for r in results}
    for tt, ng, graph in build_trivial_graphs():
        if tt not in tt_set:
            results.append((tt, ng, graph))

    print(f"Decoded {len(results)} circuits ({skipped} skipped)")

    # Sort by (n_gates, truth_table) to match prepared.npz ordering
    results.sort(key=lambda x: (x[1], x[0]))

    # Verify alignment with prepared.npz
    prepared = np.load(os.path.join(out_dir, "prepared.npz"))
    truth_tables_expected = prepared["truth_tables"]
    targets_expected = prepared["targets"]

    tt_decoded = np.array([r[0] for r in results], dtype=np.uint32)
    ng_decoded = np.array([r[1] for r in results], dtype=np.float32)

    assert len(results) == len(truth_tables_expected), \
        f"Count mismatch: {len(results)} vs {len(truth_tables_expected)}"

    tt_match = (tt_decoded == truth_tables_expected).all()
    ng_match = (ng_decoded == targets_expected).all()

    print(f"Truth table alignment: {'OK' if tt_match else 'FAIL'}")
    print(f"Gate count alignment:  {'OK' if ng_match else 'FAIL'}")

    if not tt_match:
        mismatches = np.where(tt_decoded != truth_tables_expected)[0]
        print(f"  {len(mismatches)} mismatches, first 5: {mismatches[:5]}")
        for idx in mismatches[:5]:
            print(f"    [{idx}] decoded=0x{tt_decoded[idx]:08x} expected=0x{truth_tables_expected[idx]:08x}")

    # Extract graph list
    graphs = [r[2] for r in results]

    # Stats
    num_nodes = [g.x.size(0) for g in graphs]
    num_edges = [g.edge_index.size(1) for g in graphs]
    print(f"\nGraph statistics:")
    print(f"  Nodes: min={min(num_nodes)}, max={max(num_nodes)}, "
          f"mean={np.mean(num_nodes):.1f}")
    print(f"  Edges (bidirectional): min={min(num_edges)}, max={max(num_edges)}, "
          f"mean={np.mean(num_edges):.1f}")

    # Node type distribution across all graphs
    all_types = torch.cat([g.x for g in graphs], dim=0)
    type_counts = all_types.sum(dim=0).long().tolist()
    type_names = ["const", "input", "AND", "OR", "XOR", "ANDNOT"]
    print(f"  Node type distribution:")
    for name, count in zip(type_names, type_counts):
        print(f"    {name}: {count:,}")

    # Save
    out_path = os.path.join(out_dir, "circuit_graphs.pt")
    torch.save(graphs, out_path)
    print(f"\nSaved {len(graphs)} graphs to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
