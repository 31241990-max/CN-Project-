
from dataclasses import dataclass
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
import joblib
import torch

try:
    from torch_geometric.data import Data  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    Data = None

FLOW_FEATURES = [
    "src_port","dst_port","protocol","duration","IN_BYTES","OUT_BYTES",
    "IN_PKTS","OUT_PKTS","FLOW_IAT_MEAN","FLOW_IAT_STD","TCP_FLAGS"
]

@dataclass
class GraphWindow:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray

class FlowPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, df):
        self.scaler.fit(df[FLOW_FEATURES].astype(np.float32))
        self.fitted = True
        return self

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Fit the preprocessor on training data first.")
        return self.scaler.transform(df[FLOW_FEATURES].astype(np.float32)).astype(np.float32)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        try:
            return joblib.load(path)
        except ModuleNotFoundError as exc:
            if exc.name == "data_pipeline":
                import src.data_pipeline as src_data_pipeline

                sys.modules["data_pipeline"] = src_data_pipeline
                return joblib.load(path)
            raise

def _make_node_features(num_nodes, src, dst, edge_features):
    d = edge_features.shape[1]
    sums = np.zeros((num_nodes,d), dtype=np.float32)
    counts = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(sums, src, edge_features)
    np.add.at(sums, dst, edge_features)
    np.add.at(counts, src, 1)
    np.add.at(counts, dst, 1)
    means = sums / np.maximum(counts[:,None], 1.0)
    degree = np.log1p(counts)[:,None].astype(np.float32)
    return np.concatenate([means, degree], axis=1).astype(np.float32)


def _normalize_relative_time(times):
    delta = (times - times[0]).astype(np.float32)
    delta = (delta / max(float(delta[-1]), 1e-6))[:, None]
    return delta


def build_networkx_graph(df_window, node_features, edge_features):
    graph = nx.DiGraph()
    nodes = sorted(set(df_window.src_node_id.astype(int).tolist() + df_window.dst_node_id.astype(int).tolist()))
    for node_id in nodes:
        graph.add_node(int(node_id), features=node_features[int(node_id)].astype(np.float32))

    for row_idx, (src, dst, feat) in enumerate(zip(df_window.src_node_id.to_numpy(np.int64), df_window.dst_node_id.to_numpy(np.int64), edge_features)):
        graph.add_edge(
            int(src),
            int(dst),
            edge_attr=feat.astype(np.float32),
            label=int(df_window.label.iloc[row_idx]),
            timestamp=float(df_window.FLOW_START_TIMESTAMP.iloc[row_idx]),
        )
    return graph


def make_pyg_windows(df, prep, window_size=128, stride=64, num_nodes=None):
    """Create PyG Data objects for each temporal graph snapshot."""
    if Data is None:
        raise ImportError("torch_geometric is required for PyG graph construction. Install torch-geometric.")

    df = df.sort_values("FLOW_START_TIMESTAMP").reset_index(drop=True)
    scaled = prep.transform(df)
    if num_nodes is None:
        num_nodes = int(max(df.src_node_id.max(), df.dst_node_id.max()) + 1)

    graphs = []
    for start in range(0, len(df) - window_size + 1, stride):
        w = df.iloc[start:start + window_size]
        ef = scaled[start:start + window_size]
        src = w.src_node_id.to_numpy(np.int64)
        dst = w.dst_node_id.to_numpy(np.int64)
        y = w.label.to_numpy(np.int64)
        ts = w.FLOW_START_TIMESTAMP.to_numpy(np.float64)
        delta = _normalize_relative_time(ts)
        ef = np.concatenate([ef, delta], axis=1).astype(np.float32)
        x = _make_node_features(num_nodes, src, dst, scaled[start:start + window_size])

        build_networkx_graph(w, x, ef)
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        edge_attr = torch.tensor(ef, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.long)
        timestamps = torch.tensor(ts, dtype=torch.float32)
        node_attr = torch.tensor(x, dtype=torch.float32)

        pyg_data = Data(
            x=node_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels,
            timestamps=timestamps,
            num_nodes=num_nodes,
        )
        graphs.append(pyg_data)
    return graphs


def make_pyg_dataloader(df, prep, window_size=128, stride=64, batch_size=32, shuffle=False, num_nodes=None):
    """Return a PyG DataLoader that batches temporal graph snapshots."""
    try:
        from torch_geometric.loader import DataLoader  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch_geometric is required to build a graph DataLoader.") from exc

    graphs = make_pyg_windows(df, prep, window_size=window_size, stride=stride, num_nodes=num_nodes)
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)


def make_windows(df, prep, window_size=128, stride=64, num_nodes=None):
    df = df.sort_values("FLOW_START_TIMESTAMP").reset_index(drop=True)
    scaled = prep.transform(df)
    if num_nodes is None:
        num_nodes = int(max(df.src_node_id.max(), df.dst_node_id.max()) + 1)
    windows=[]
    for start in range(0, len(df)-window_size+1, stride):
        w=df.iloc[start:start+window_size]
        ef=scaled[start:start+window_size]
        src=w.src_node_id.to_numpy(np.int64)
        dst=w.dst_node_id.to_numpy(np.int64)
        y=w.label.to_numpy(np.int64)
        ts=w.FLOW_START_TIMESTAMP.to_numpy(np.float64)
        delta=(ts-ts[0]).astype(np.float32)
        delta=(delta/max(float(delta[-1]),1e-6))[:,None]
        ef=np.concatenate([ef,delta],axis=1).astype(np.float32)
        x=_make_node_features(num_nodes,src,dst,scaled[start:start+window_size])
        windows.append(GraphWindow(x,np.stack([src,dst]),ef,y,ts))
    return windows


def make_prefix_windows(df, prep, window_size=128, stride=64, num_nodes=None, prefix_fractions=(0.1, 0.2, 0.3, 0.5)):
    """Augment training data with prefix-only graph snapshots to improve early attack detection."""
    base_windows = make_windows(df, prep, window_size=window_size, stride=stride, num_nodes=num_nodes)
    windows = []
    for base in base_windows:
        windows.append(base)
        for frac in prefix_fractions:
            if frac <= 0 or frac > 1:
                continue
            prefix_len = max(1, int(base.edge_features.shape[0] * frac))
            src = base.edge_index[0, :prefix_len]
            dst = base.edge_index[1, :prefix_len]
            prefix_features = base.edge_features[:prefix_len]
            prefix_labels = base.labels[:prefix_len]
            prefix_ts = base.timestamps[:prefix_len]
            prefix_x = _make_node_features(num_nodes or int(max(df.src_node_id.max(), df.dst_node_id.max()) + 1), src, dst, prefix_features[:, :-1])
            windows.append(GraphWindow(prefix_x, np.stack([src, dst]), prefix_features, prefix_labels, prefix_ts))
    return windows


def save_metadata(path,num_nodes,edge_dim,window_size,stride,decision_threshold=0.5):
    Path(path).write_text(json.dumps({
        "num_nodes":int(num_nodes),"edge_feature_dim":int(edge_dim),
        "window_size":int(window_size),"stride":int(stride),
        "decision_threshold":float(decision_threshold),
        "label_mapping":{"0":"Benign","1":"Attack"}
    },indent=2))
