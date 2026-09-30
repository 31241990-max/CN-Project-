import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from data_pipeline import FlowPreprocessor, make_windows
from model import HybridGNNTransformer

FRACTIONS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def load_assets(out_dir="outputs", data_dir="processed"):
    out = Path(out_dir)
    meta = json.loads((out / "metadata.json").read_text())
    ckpt = torch.load(out / "best_model.pt", map_location="cpu")
    prep = FlowPreprocessor.load(out / "scaler.joblib")
    df = pd.read_csv(Path(data_dir) / "test.csv")

    model = HybridGNNTransformer(
        ckpt["node_dim"], ckpt["edge_dim"], max_seq_len=meta["window_size"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    windows = make_windows(
        df,
        prep,
        meta["window_size"],
        meta["stride"],
        meta["num_nodes"],
    )
    return model, windows, float(ckpt.get("decision_threshold", meta.get("decision_threshold", 0.5)))


def _make_prefix_node_features(num_nodes, edge_index, edge_features):
    src = edge_index[0]
    dst = edge_index[1]
    base_features = edge_features[:, :-1]
    sums = np.zeros((num_nodes, base_features.shape[1]), dtype=np.float32)
    counts = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(sums, src, base_features)
    np.add.at(sums, dst, base_features)
    np.add.at(counts, src, 1)
    np.add.at(counts, dst, 1)
    means = sums / np.maximum(counts[:, None], 1.0)
    degree = np.log1p(counts)[:, None].astype(np.float32)
    return torch.from_numpy(np.concatenate([means, degree], axis=1).astype(np.float32))


def partial_window(window, fraction, num_nodes):
    n = max(1, int(window.edge_features.shape[0] * fraction))
    node_x = _make_prefix_node_features(num_nodes, window.edge_index[:, :n], window.edge_features[:n])
    edge_index = torch.from_numpy(window.edge_index[:, :n]).long()
    edge_features = torch.from_numpy(window.edge_features[:n]).float()
    labels = torch.from_numpy(window.labels[:n]).long()
    return node_x, edge_index, edge_features, labels


@torch.no_grad()
def evaluate_fraction(model, window, fraction, num_nodes, threshold):
    node_x, edge_index, edge_features, labels = partial_window(window, fraction, num_nodes)
    logits = model(node_x, edge_index, edge_features)
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    y_true = labels.cpu().numpy()
    y_pred = (probs >= threshold).astype(int)
    has_both_classes = len(np.unique(y_true)) == 2

    return {
        "fraction": fraction,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs) if has_both_classes else None,
        "pr_auc": average_precision_score(y_true, probs) if has_both_classes else None,
        "threshold": threshold,
    }


def evaluate_early_detection(model, windows, num_nodes, threshold):
    results = []
    for fraction in FRACTIONS:
        all_metrics = [evaluate_fraction(model, w, fraction, num_nodes, threshold) for w in windows]
        roc_values = [m["roc_auc"] for m in all_metrics if m["roc_auc"] is not None]
        pr_values = [m["pr_auc"] for m in all_metrics if m["pr_auc"] is not None]
        results.append(
            {
                "fraction": fraction,
                "precision": float(np.mean([m["precision"] for m in all_metrics])),
                "recall": float(np.mean([m["recall"] for m in all_metrics])),
                "f1": float(np.mean([m["f1"] for m in all_metrics])),
            "roc_auc": float(np.mean(roc_values)) if roc_values else None,
            "pr_auc": float(np.mean(pr_values)) if pr_values else None,
            "threshold": threshold,
            }
        )
    return results


def plot_tradeoff(results, out_dir="outputs"):
    x = [r["fraction"] for r in results]
    y_f1 = [r["f1"] for r in results]
    y_recall = [r["recall"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_f1, marker="o", label="Attack F1")
    plt.plot(x, y_recall, marker="s", label="Attack Recall")
    plt.xlabel("Observed sequence fraction")
    plt.ylabel("Detection performance")
    plt.title("Time-to-Detection vs. Accuracy")
    plt.xticks(x)
    plt.grid(True)
    plt.legend()
    out_path = Path(out_dir).resolve() / "early_detection_curve.png"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(str(out_path))
    plt.close()
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--data-dir", default="processed")
    args = parser.parse_args()
    model, windows, threshold = load_assets(args.out_dir, args.data_dir)
    num_nodes = max(max(w.node_features.shape[0] for w in windows), 1)
    results = evaluate_early_detection(model, windows, num_nodes, threshold)
    print("Early detection metrics by observation fraction:")
    for r in results:
        print(r)
    plot_tradeoff(results, args.out_dir)


if __name__ == "__main__":
    main()
