import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_pipeline import FlowPreprocessor, make_windows
from model import HybridGNNTransformer


def stream_rows(csv_path, batch_size=1, max_rows=None, shuffle=False):
    """Yield rows like a live network stream from a static CSV."""
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.iloc[:max_rows].copy()
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    current = []
    for _, row in df.iterrows():
        current.append(row)
        if len(current) >= batch_size:
            yield current
            current = []
    if current:
        yield current


def build_rolling_buffer(window_size, maxlen=None):
    if maxlen is None:
        maxlen = window_size
    return deque(maxlen=maxlen)


def _make_node_features(num_nodes, src, dst, edge_features):
    d = edge_features.shape[1]
    sums = np.zeros((num_nodes, d), dtype=np.float32)
    counts = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(sums, src, edge_features)
    np.add.at(sums, dst, edge_features)
    np.add.at(counts, src, 1)
    np.add.at(counts, dst, 1)
    means = sums / np.maximum(counts[:, None], 1.0)
    degree = np.log1p(counts)[:, None].astype(np.float32)
    return np.concatenate([means, degree], axis=1).astype(np.float32)


def snapshot_from_rows(rows, prep, num_nodes, window_size=None):
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.sort_values("FLOW_START_TIMESTAMP").reset_index(drop=True)
    scaled = prep.transform(df)
    if window_size is not None:
        scaled = scaled[:window_size]
        df = df.iloc[:window_size].reset_index(drop=True)

    src = df["src_node_id"].to_numpy(np.int64)
    dst = df["dst_node_id"].to_numpy(np.int64)
    labels = df["label"].to_numpy(np.int64)
    ts = df["FLOW_START_TIMESTAMP"].to_numpy(np.float64)
    delta = (ts - ts[0]).astype(np.float32)
    if len(delta) > 1:
        delta = (delta / max(float(delta[-1]), 1e-6))[:, None]
    else:
        delta = np.zeros((len(ts), 1), dtype=np.float32)
    edge_features = np.concatenate([scaled, delta], axis=1).astype(np.float32)
    node_features = _make_node_features(num_nodes, src, dst, scaled)
    return {
        "node_features": node_features,
        "edge_index": np.stack([src, dst]),
        "edge_features": edge_features,
        "labels": labels,
        "timestamps": ts,
    }


def load_model_and_prep():
    out = Path("outputs")
    meta = json.loads((out / "metadata.json").read_text())
    ckpt = torch.load(out / "best_model.pt", map_location="cpu")
    prep = FlowPreprocessor.load(out / "scaler.joblib")
    model = HybridGNNTransformer(
        ckpt["node_dim"], ckpt["edge_dim"], max_seq_len=meta["window_size"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, prep, meta


def simulate_live_stream(csv_path, window_size=128, num_nodes=None, max_rows=None):
    model, prep, meta = load_model_and_prep()
    if num_nodes is None:
        num_nodes = meta["num_nodes"]

    buffer = build_rolling_buffer(window_size)
    stream = stream_rows(csv_path, batch_size=1, max_rows=max_rows)
    outputs = []

    for batch in stream:
        row = batch[0]
        buffer.append(row)
        if len(buffer) == 0:
            continue

        snapshot = snapshot_from_rows(list(buffer), prep, num_nodes, window_size)
        if snapshot is None:
            continue

        node_x = torch.from_numpy(snapshot["node_features"]).float()
        edge_index = torch.from_numpy(snapshot["edge_index"]).long()
        edge_attr = torch.from_numpy(snapshot["edge_features"]).float()
        labels = torch.from_numpy(snapshot["labels"]).long()

        start = time.perf_counter()
        with torch.no_grad():
            logits = model(node_x, edge_index, edge_attr)
            probs = torch.softmax(logits, dim=1)[:, 1]
        latency = time.perf_counter() - start

        pred = int((probs[-1].item() >= 0.5))
        outputs.append(
            {
                "timestamp": float(snapshot["timestamps"][-1]),
                "probability": float(probs[-1].item()),
                "prediction": pred,
                "true_label": int(labels[-1].item()),
                "latency_seconds": latency,
            }
        )

    return outputs


def profile_latency(csv_path, window_size=128, samples=100, num_nodes=None):
    model, prep, meta = load_model_and_prep()
    if num_nodes is None:
        num_nodes = meta["num_nodes"]

    rows = pd.read_csv(csv_path).head(samples).to_dict(orient="records")
    latencies = []
    for i in range(len(rows)):
        window = rows[max(0, i - window_size + 1): i + 1]
        snapshot = snapshot_from_rows(window, prep, num_nodes, window_size)
        if snapshot is None:
            continue
        node_x = torch.from_numpy(snapshot["node_features"]).float()
        edge_index = torch.from_numpy(snapshot["edge_index"]).long()
        edge_attr = torch.from_numpy(snapshot["edge_features"]).float()

        start = time.perf_counter()
        with torch.no_grad():
            model(node_x, edge_index, edge_attr)
        latencies.append(time.perf_counter() - start)

    return {
        "sample_count": len(latencies),
        "mean_latency_seconds": float(np.mean(latencies)),
        "p95_latency_seconds": float(np.percentile(latencies, 95)),
        "max_latency_seconds": float(np.max(latencies)),
        "min_latency_seconds": float(np.min(latencies)),
    }


if __name__ == "__main__":
    csv_path = "processed/test.csv"
    stream_results = simulate_live_stream(csv_path, window_size=128)
    print("Live stream preview:")
    for item in stream_results[:5]:
        print(item)

    stats = profile_latency(csv_path, window_size=128, samples=50)
    print("\nInference latency profile:")
    print(stats)
