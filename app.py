import json
import hashlib
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from src.data_pipeline import FlowPreprocessor
from src.model import HybridGNNTransformer


@st.cache_resource
def load_assets():
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


def build_snapshot(df_window, prep, num_nodes):
    if df_window.empty:
        return None
    df = df_window.sort_values("FLOW_START_TIMESTAMP").reset_index(drop=True)
    scaled = prep.transform(df)
    ts = df["FLOW_START_TIMESTAMP"].to_numpy(np.float64)
    delta = (ts - ts[0]).astype(np.float32)
    if len(delta) > 1:
        delta = (delta / max(float(delta[-1]), 1e-6))[:, None]
    else:
        delta = np.zeros((len(ts), 1), dtype=np.float32)
    edge_features = np.concatenate([scaled, delta], axis=1).astype(np.float32)
    source_ids = df["src_node_id"].to_numpy(np.int64)
    destination_ids = df["dst_node_id"].to_numpy(np.int64)
    active_ids = np.unique(np.concatenate([source_ids, destination_ids]))
    src = np.searchsorted(active_ids, source_ids).astype(np.int64)
    dst = np.searchsorted(active_ids, destination_ids).astype(np.int64)
    node_features = _make_node_features(len(active_ids), src, dst, scaled)
    return {
        "node_features": node_features,
        "edge_index": np.stack([src, dst]),
        "edge_features": edge_features,
        "labels": df["label"].to_numpy(np.int64),
        "timestamps": ts,
    }


def predict_window(model, prep, meta, df_window):
    snapshot = build_snapshot(df_window, prep, meta["num_nodes"])
    if snapshot is None:
        return None, None, None

    node_x = torch.from_numpy(snapshot["node_features"]).float()
    edge_index = torch.from_numpy(snapshot["edge_index"]).long()
    edge_attr = torch.from_numpy(snapshot["edge_features"]).float()

    with torch.no_grad():
        logits, explanations = model.forward_with_explanations(
            node_x, edge_index, edge_attr
        )
        prob = torch.softmax(logits, 1)[:, 1].cpu().numpy()
    attention = explanations["attention"].mean(dim=0)[-1].cpu().numpy()
    gcn_weights = explanations["gcn_edge_weights"].mean(dim=0).cpu().numpy()
    return prob, snapshot["timestamps"], snapshot["labels"], attention, gcn_weights


def _stable_position(ip_address):
    digest = hashlib.blake2b(str(ip_address).encode(), digest_size=8).digest()
    angle = int.from_bytes(digest[:4], "big") / 2**32 * 2 * np.pi
    radius = 0.35 + int.from_bytes(digest[4:], "big") / 2**32 * 0.65
    return radius * np.cos(angle), radius * np.sin(angle)


def build_topology_figure(df, probabilities, gcn_weights, threshold):
    df = df.sort_values("FLOW_START_TIMESTAMP").reset_index(drop=True)
    graph = nx.DiGraph()
    edge_groups = {}
    for index, row in df.iterrows():
        src_ip = str(row.get("src_ip", row["src_node_id"]))
        dst_ip = str(row.get("dst_ip", row["dst_node_id"]))
        probability = float(probabilities[index])
        propagation = float(gcn_weights[index])
        graph.add_edge(src_ip, dst_ip)
        edge = edge_groups.setdefault((src_ip, dst_ip), {
            "probability": probability,
            "propagation": [],
            "count": 0,
            "row": row,
        })
        edge["probability"] = max(edge["probability"], probability)
        edge["propagation"].append(propagation)
        edge["count"] += 1
        edge["row"] = row

    figure = go.Figure()
    for (src_ip, dst_ip), edge in edge_groups.items():
        probability = edge["probability"]
        propagation = float(np.mean(edge["propagation"]))
        row = edge["row"]
        src_x, src_y = _stable_position(src_ip)
        dst_x, dst_y = _stable_position(dst_ip)
        malicious = probability >= threshold
        figure.add_trace(go.Scatter(
            x=[src_x, dst_x], y=[src_y, dst_y], mode="lines",
            line={"color": "#d64b4b" if malicious else "#72939a",
                  "width": 1.5 + 4.5 * probability},
            opacity=0.82,
            text=[
                f"{src_ip} → {dst_ip}<br>Attack probability: {probability:.3f}"
                f"<br>GCN propagation weight: {propagation:.4f}"
                f"<br>Flows in window: {edge['count']}"
                f"<br>Timestamp: {row['FLOW_START_TIMESTAMP']}"
            ] * 2,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    labelled_nodes = set(sorted(graph.nodes, key=graph.degree, reverse=True)[:15])
    node_x, node_y, node_text, node_color, node_size, node_labels = [], [], [], [], [], []
    for ip_address in graph.nodes:
        x, y = _stable_position(ip_address)
        node_x.append(x)
        node_y.append(y)
        degree = graph.degree(ip_address)
        flagged = any(
            (src == ip_address or dst == ip_address)
            and edge["probability"] >= threshold
            for (src, dst), edge in edge_groups.items()
        )
        node_text.append(f"{ip_address}<br>Connections: {degree}")
        node_labels.append(ip_address if ip_address in labelled_nodes else "")
        node_color.append("#d64b4b" if flagged else "#2f8f83")
        node_size.append(12 + min(degree, 16))

    figure.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_labels, textposition="top center",
        hovertext=node_text, hovertemplate="%{hovertext}<extra></extra>",
        marker={"size": node_size, "color": node_color,
                "line": {"width": 1.5, "color": "#ffffff"}},
        showlegend=False,
    ))
    figure.update_layout(
        height=520, margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False, "fixedrange": True},
        yaxis={"visible": False, "fixedrange": True, "scaleanchor": "x"},
        hovermode="closest",
    )
    return figure


def build_attention_figure(df, attention, probabilities, threshold):
    df = df.sort_values("FLOW_START_TIMESTAMP").reset_index(drop=True)
    top_indices = np.argsort(attention)[-12:]
    top_indices = top_indices[np.argsort(attention[top_indices])]
    timestamps = df["FLOW_START_TIMESTAMP"].to_numpy(np.float64)
    inter_arrival = np.diff(timestamps, prepend=timestamps[0])
    labels, details, colors = [], [], []
    for index in top_indices:
        row = df.iloc[index]
        src_ip = str(row.get("src_ip", row["src_node_id"]))
        dst_ip = str(row.get("dst_ip", row["dst_node_id"]))
        labels.append(f"#{index + 1} {src_ip} → {dst_ip}")
        details.append(
            f"Attention: {attention[index]:.4f}<br>"
            f"Inter-arrival: {inter_arrival[index]:.6g}<br>"
            f"Attack probability: {probabilities[index]:.3f}<br>"
            f"Timestamp: {timestamps[index]}"
        )
        colors.append("#d64b4b" if probabilities[index] >= threshold else "#2f8f83")

    figure = go.Figure(go.Bar(
        x=attention[top_indices], y=labels, orientation="h",
        marker_color=colors, customdata=details,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    figure.update_layout(
        height=520, margin={"l": 12, "r": 12, "t": 8, "b": 8},
        xaxis_title="Mean attention from the latest packet",
        yaxis={"automargin": True},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


st.set_page_config(page_title="Cyber Threat Monitor", layout="wide")
st.title("Cyber Threat Monitoring Dashboard")

model, prep, meta = load_assets()

with st.sidebar:
    st.header("Configuration")
    default_threshold = min(0.99, max(0.01, float(meta.get("decision_threshold", 0.5))))
    threshold = st.slider("Threat detection threshold", 0.01, 0.99, default_threshold, step=0.01)
    window_size = st.number_input(
        "Window size", min_value=8, max_value=max(8, int(meta["window_size"])),
        value=min(128, max(8, int(meta["window_size"]))), step=8,
    )
    refresh_seconds = st.slider("Refresh interval (seconds)", 0.5, 5.0, 1.0, step=0.5)
    st.caption("The model evaluates the most recent stream window and updates the dashboard in real time.")

stream_df = pd.read_csv("processed/test.csv")


@st.fragment(run_every=refresh_seconds)
def render_live_dashboard():
    current_index = st.session_state.get("current_index", 0)
    alerts = st.session_state.setdefault("alerts", [])
    now = time.perf_counter()
    previous_tick = st.session_state.get("previous_rate_tick")
    replay_rate = 0.0 if previous_tick is None else 1.0 / max(now - previous_tick, 1e-6)
    st.session_state.previous_rate_tick = now
    if len(stream_df) == 0:
        st.info("Waiting for stream data...")
        return

    start_idx = max(0, current_index - window_size + 1)
    recent_df = stream_df.iloc[start_idx:current_index + 1].copy()
    recent_df = recent_df.sort_values("FLOW_START_TIMESTAMP").reset_index(drop=True)

    if recent_df.empty:
        st.session_state.current_index = 0
        return

    prob, timestamps, labels, attention, gcn_weights = predict_window(
        model, prep, meta, recent_df
    )

    if prob is not None:
        latest_prob = float(prob[-1])
        latest_label = int(labels[-1]) if len(labels) > 0 else 0
        threat_level = "HIGH" if latest_prob >= threshold else "LOW"
        if current_index >= len(stream_df) - 1:
            current_index = 0
        else:
            current_index += 1

        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Current Threat Level", threat_level)
        with metric_cols[1]:
            st.metric("Active IP Nodes", str(len(set(recent_df["src_node_id"]).union(set(recent_df["dst_node_id"]))) ))
        with metric_cols[2]:
            st.metric(
                "Replay Ingest Rate / Sec", f"{replay_rate:.1f}",
                help="One CSV flow is advanced per dashboard refresh; rate uses elapsed wall-clock time.",
            )
        with metric_cols[3]:
            st.metric("Latest Attack Probability", f"{latest_prob:.3f}")

        topology_col, attention_col = st.columns([1.15, 0.85])
        with topology_col:
            st.subheader("Network topology")
            st.plotly_chart(
            build_topology_figure(recent_df, prob, gcn_weights, threshold),
                width="stretch", key="network-topology",
            )
        with attention_col:
            st.subheader("Packets attended to by latest prediction")
            st.caption("Attention indicates model context, not causal proof. Hover for inter-arrival timing and GCN propagation weight.")
            st.plotly_chart(
            build_attention_figure(recent_df, attention, prob, threshold),
                width="stretch", key="packet-attention",
            )

        with st.container():
            if latest_prob >= threshold:
                alert_msg = f"ALERT: Attack probability {latest_prob:.3f} at {timestamps[-1]}"
                last_alert_index = st.session_state.get("last_alert_index")
                if last_alert_index != current_index:
                    alerts.append({"time": timestamps[-1], "prob": latest_prob, "row": recent_df.iloc[-1].to_dict()})
                    st.session_state.last_alert_index = current_index
                    if len(alerts) > 5:
                        alerts.pop(0)
                st.error(alert_msg)
                for alert in reversed(alerts[-5:]):
                    st.write(f"{alert['time']} | src={alert['row']['src_ip']} | dst={alert['row']['dst_ip']} | prob={alert['prob']:.3f}")
            else:
                st.info("No active anomaly detected in the current window.")
    else:
        st.info("Waiting for stream data...")

    st.session_state.current_index = current_index


render_live_dashboard()
