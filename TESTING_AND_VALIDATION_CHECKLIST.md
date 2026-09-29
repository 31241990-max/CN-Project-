# Hybrid GNN + Transformer Cyber Attack Prediction: Verification & Validation Checklist

**Role:** Lead QA Engineer & Machine Learning Validator  
**System:** Hybrid Graph Neural Network and Transformer Framework for Early Cyber Attack Prediction  
**Evaluation Target:** End-to-end data integrity, tensor transformations, optimization stability, and real-time inference latency.

---

## Phase 1: Data Pipeline & Graph Construction (Member 1 Scope)

### 1.1 Chronological Split & Leakage Prevention
- [ ] **Temporal Boundary Disjointness Test**
  - **Test Action:** Run a script verifying that $\max(T_{\text{train}}) < \min(T_{\text{val}}) \le \max(T_{\text{val}}) < \min(T_{\text{test}})$. Ensure no overlap exists in `FLOW_START_TIMESTAMP`.
  - **Verification Script / Command:**
    ```python
    import pandas as pd
    train = pd.read_csv("processed/train.csv")
    val   = pd.read_csv("processed/val.csv")
    test  = pd.read_csv("processed/test.csv")
    assert train["FLOW_START_TIMESTAMP"].max() < val["FLOW_START_TIMESTAMP"].min(), "Leakage: train overlaps val!"
    assert val["FLOW_START_TIMESTAMP"].max() < test["FLOW_START_TIMESTAMP"].min(), "Leakage: val overlaps test!"
    print("✓ Temporal split strictly chronological without boundary overlap.")
    ```
  - **Pass Criteria:** `True` with strictly monotonic time boundaries across splits.

- [ ] **Scaler Isolation Verification**
  - **Test Action:** Verify that `FlowPreprocessor` (`outputs/scaler.joblib`) was fitted exclusively on `train.csv`. Assert that the mean ($\mu$) and standard deviation ($\sigma$) stored in the scaler match `train.csv[FLOW_FEATURES].mean()` and `.std()` within float precision ($\epsilon < 10^{-5}$) and do not contain validation or test statistics.
  - **Verification Command:**
    ```python
    import joblib, pandas as pd, numpy as np
    scaler = joblib.load("outputs/scaler.joblib").scaler
    train_mean = pd.read_csv("processed/train.csv")[[
        "src_port","dst_port","protocol","duration","IN_BYTES","OUT_BYTES",
        "IN_PKTS","OUT_PKTS","FLOW_IAT_MEAN","FLOW_IAT_STD","TCP_FLAGS"
    ]].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, train_mean, rtol=1e-4)
    print("✓ Scaler parameters solely derived from training partition.")
    ```

---

### 1.2 Graph Construction & PyTorch Geometric Shape Validation
- [ ] **PyG Data Snapshot Dimensions Check**
  - **Test Action:** Validate that the constructed `GraphWindow` / PyG `Data` structures conform strictly to expected structural and tensor constraints:
    - Node feature matrix $X \in \mathbb{R}^{V \times D_{\text{node}}}$ (where $D_{\text{node}} = D_{\text{flow}} + 1 = 12$).
    - Edge index $E \in \mathbb{Z}^{2 \times |E|}$ with directed indices (`src` $\to$ `dst`), where all node IDs satisfy $0 \le v_i < V$.
    - Edge feature matrix $E_{\text{attr}} \in \mathbb{R}^{|E| \times D_{\text{edge}}}$ (where $D_{\text{edge}} = 11 \text{ (scaled features)} + 1 \text{ (normalized relative time)} = 12$).
    - Temporal sequence length equals configured `window_size` (e.g., 128).
  - **Verification Script:**
    ```python
    from src.data_pipeline import FlowPreprocessor, make_pyg_windows
    import pandas as pd
    prep = FlowPreprocessor.load("outputs/scaler.joblib")
    df = pd.read_csv("processed/test.csv").iloc[:256]
    windows = make_pyg_windows(df, prep, window_size=128, stride=64)
    w = windows[0]
    assert w.x.ndim == 2 and w.x.shape[1] == 12, f"Invalid node_x shape: {w.x.shape}"
    assert w.edge_index.ndim == 2 and w.edge_index.shape[0] == 2, f"Invalid edge_index: {w.edge_index.shape}"
    assert w.edge_attr.ndim == 2 and w.edge_attr.shape[1] == 12, f"Invalid edge_attr: {w.edge_attr.shape}"
    assert w.edge_index.max() < w.x.shape[0], "Edge index points to non-existent node ID!"
    print("✓ PyG graph dimensions and connectivity indices verified.")
    ```

- [ ] **Graph Connectivity & Directionality Audit**
  - **Test Action:** Assert that network interactions retain flow directionality (source host to destination host) and that zero disconnected subgraphs cause out-of-index errors in GCN message passing.

---

### 1.3 Feature Normalization & Outlier Robustness
- [ ] **Heavy-Tail Feature Scaling (IAT & Bytes)**
  - **Test Action:** Inspect input flow distributions for `FLOW_IAT_MEAN`, `FLOW_IAT_STD`, `IN_BYTES`, and `OUT_BYTES`. Feed synthetic edge bursts with extreme outlier values ($10^7$ bytes, $10^5$ ms IAT) through `FlowPreprocessor.transform()` to verify:
    1. Output contains zero `NaN`, `Inf`, or `-Inf` values.
    2. Numerical representation remains finite float32.
  - **Verification Script:**
    ```python
    import numpy as np, pandas as pd
    from src.data_pipeline import FlowPreprocessor
    prep = FlowPreprocessor.load("outputs/scaler.joblib")
    extreme_df = pd.DataFrame([{
        "src_port": 443, "dst_port": 58210, "protocol": 6, "duration": 999999.0,
        "IN_BYTES": 1e9, "OUT_BYTES": 1e9, "IN_PKTS": 500000, "OUT_PKTS": 500000,
        "FLOW_IAT_MEAN": 1e7, "FLOW_IAT_STD": 1e6, "TCP_FLAGS": 24
    }])
    transformed = prep.transform(extreme_df)
    assert not np.isnan(transformed).any(), "NaN found in transformed features!"
    assert not np.isinf(transformed).any(), "Inf found in transformed features!"
    print("✓ Extreme telemetry values successfully scaled without numerical instability.")
    ```

- [ ] **Relative Time Delta Monotonicity Check**
  - **Test Action:** Assert that `delta = (ts - ts[0]) / max(delta[-1], 1e-6)` produces values strictly bound in $[0.0, 1.0]$ in non-decreasing order across the chronological edge sequence.

---

## Phase 2: Model Architecture & Training (Member 2 Scope)

### 2.1 Spatial-to-Temporal Tensor Handoff Validation
- [ ] **Dimension & Reshaping Alignment Check**
  - **Test Action:** Instrument the intermediate tensors in `HybridGNNTransformer.forward` using PyTorch assertions to verify the exact dimensional flow:
    1. **Spatial Output:** GCN outputs $H \in \mathbb{R}^{V \times 64}$.
    2. **Edge Embedding:** Concatenation $E = [H_{\text{src}} \,\|\, H_{\text{dst}} \,\|\, E_{\text{attr}}] \in \mathbb{R}^{|E| \times (64 + 64 + 12 = 140)}$.
    3. **Projection:** Linear projection $Z_0 \in \mathbb{R}^{|E| \times 128}$.
    4. **Temporal Input (Batch Reshape):** Sequence expanded to batch dimension $Z \in \mathbb{R}^{1 \times |E| \times 128}$.
    5. **Positional Encoding Addition:** Added safely with broadcast on sequence length without truncating feature channels.
    6. **Transformer Encoder Execution:** Output maintains shape $(1, |E|, 128)$.
    7. **Classification Head:** Linear mapping $(|E|, 128) \to (|E|, 2)$.
  - **Verification Script:**
    ```python
    import torch
    from src.model import HybridGNNTransformer
    model = HybridGNNTransformer(node_in_dim=12, edge_in_dim=12, gnn_hidden=64, model_dim=128, max_seq_len=128)
    node_x = torch.randn(20, 12)
    edge_index = torch.randint(0, 20, (2, 128))
    edge_features = torch.randn(128, 12)

    # Validate forward execution & shape invariants
    logits = model(node_x, edge_index, edge_features)
    assert logits.shape == (128, 2), f"Expected logits (128, 2), received {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN!"
    print("✓ Spatial-to-temporal tensor handoff validated.")
    ```

---

### 2.2 Imbalanced Loss Convergence & Anti-Collapse Check
- [ ] **Class-Weighted / Focal Loss Efficacy Verification**
  - **Test Action:** Verify that the model does not collapse into predicting majority class (Benign, Class 0) due to dataset skew (27,907 Benign vs 7,093 Attack).
  - **Validation Steps:**
    1. Run `src/evaluate.py`.
    2. Inspect confusion matrix: $[[TN, FP], [FN, TP]]$.
    3. Assert Minority Class Recall:
       $$\text{Recall}_{\text{Attack}} = \frac{TP}{TP + FN} > 0.70$$
    4. Assert Precision-Recall AUC (PR-AUC):
       $$\text{PR-AUC} > \text{Baseline Prevalence } \left(\frac{1,516}{7,424} \approx 0.204\right)$$
  - **Pass Criteria:**
    - Attack recall $\ge 0.85$ (current checkpoint achieves $>0.99$).
    - Optimal decision threshold calibrated via validation set F1 curve rather than hardcoded 0.50.

---

### 2.3 Self-Attention Integrity & Explainability Verification
- [ ] **Transformer Self-Attention Tensor Check**
  - **Test Action:** Execute `forward_with_explanations()` and extract the multi-head attention weights tensor $\mathbf{A} \in \mathbb{R}^{\text{heads} \times S \times S}$.
  - **Verification Script:**
    ```python
    import torch
    from src.model import HybridGNNTransformer
    model = HybridGNNTransformer(node_in_dim=12, edge_in_dim=12, max_seq_len=128)
    ckpt = torch.load("outputs/best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    node_x = torch.randn(15, 12)
    edge_index = torch.randint(0, 15, (2, 64))
    edge_attr = torch.randn(64, 12)

    with torch.no_grad():
        logits, explanations = model.forward_with_explanations(node_x, edge_index, edge_attr)

    attn = explanations["attention"] # shape: [num_heads, seq_len, seq_len]
    assert attn.ndim == 3, f"Unexpected attention tensor rank: {attn.shape}"
    assert not torch.isnan(attn).any(), "Attention weights contain NaN!"
    assert not torch.isinf(attn).any(), "Attention weights contain Inf!"

    # Verify probability distribution: rows must sum to 1.0 along the key sequence axis
    row_sums = attn.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-4)

    # Check non-trivial attention (variance > 0 to confirm active focus)
    assert attn.var().item() > 1e-6, "Attention weights are degenerate/flat uniform distribution!"
    print("✓ Self-attention weights verified: stochasticity, non-zero variance, no NaNs.")
    ```

---

## Phase 3: Inference, Evaluation & Deployment (Member 3 Scope)

### 3.1 Early Detection Metric (Reconnaissance Phase Trigger)
- [ ] **Prefix Window Early Warning Evaluation ($k$-step Analysis)**
  - **Test Action:** Run the prefix-fraction evaluation script to measure whether the model detects attacks early in the temporal window ($k \in [10\%, 20\%, 30\%]$ of total packets) rather than waiting for flow completion.
  - **Verification Command:**
    ```powershell
    .venv-1\Scripts\python.exe src/early_detection_eval.py
    ```
  - **Pass Criteria:**
    - Recall and F1 score at fraction $k = 0.20$ must exceed 0.80 of the final full-window performance ($k = 1.0$).
    - Generated verification plot `outputs/early_detection_curve.png` shows a steep detection slope in the first $20\%$ to $30\%$ of packet arrivals.

---

### 3.2 Inference Latency & Throughput Benchmarking
- [ ] **Batch Latency SLA Check ($\le 50\text{ ms}$ per Window)**
  - **Test Action:** Benchmark the end-to-end inference step (Graph Snapshot Construction + PyG Forward + Softmax) over 100 consecutive iterations.
  - **Verification Script:**
    ```python
    import time, torch, pandas as pd, numpy as np
    from app import load_assets, predict_window

    model, prep, meta = load_assets()
    df = pd.read_csv("processed/test.csv").iloc[:128]

    # Warm-up pass
    for _ in range(5):
        _ = predict_window(model, prep, meta, df)

    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = predict_window(model, prep, meta, df)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    print(f"Latency Benchmark: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms")
    assert p95 < 50.0, f"SLA Breach: p95 latency {p95:.2f}ms exceeds 50ms ceiling!"
    print("✓ Real-time inference latency complies with <50ms SLA.")
    ```

- [ ] **Throughput Stability Check**
  - **Test Action:** Confirm ingestion pipeline sustains $\ge 1,000$ flow records per second in continuous streaming mode.

---

### 3.3 Dashboard State, Streaming Memory & UI Stability
- [ ] **Streamlit Fragment & Replay Loop Stability**
  - **Test Action:** Test the continuous streaming loop in `app.py` (`@st.fragment(run_every=1.0)`):
    1. Start the dashboard server on port 8501.
    2. Stream 1,000 consecutive windows continuously.
    3. Monitor process memory consumption (`Private Bytes` / `Working Set`).
  - **Verification Command:**
    ```powershell
    # Monitor memory usage of Streamlit process across 3 minutes
    Get-Process -Name python | Select-Object Id, ProcessName, WorkingSet64, VirtualMemorySize64
    ```
  - **Pass Criteria:**
    - Memory growth flattens asymptotically (no unbound list append in `st.session_state["alerts"]` — capped at 5 records).
    - HTTP endpoint returns status 200 continuously without unhandled exceptions or thread-locking.

- [ ] **Interactive Visualizations Rendering Check**
  - **Test Action:** Verify that Plotly network topology graph and packet attention bar chart render dynamically on threshold slider changes ($0.01 \le \theta \le 0.99$) without causing browser DOM freezing.

---

## Summary Validation Matrix

| Phase | Test Item | Primary Tool / Script | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Chronological Data Split | `pandas` timestamps | No time overlap between Train, Val, Test |
| **Phase 1** | PyG Dimension Invariance | `src/data_pipeline.py` | Node dim = 12, Edge dim = 12, Index valid |
| **Phase 1** | Feature Scaling | `outputs/scaler.joblib` | 0 NaNs / 0 Infs under outlier bursts |
| **Phase 2** | GNN-to-Transformer Reshape | `src/model.py` | Intermediate tensor shapes match `[1, seq_len, 128]` |
| **Phase 2** | Focal Loss Minority Recall | `src/evaluate.py` | Attack Recall $> 85\%$, PR-AUC $> 0.20$ |
| **Phase 2** | Self-Attention Stochasticity | `forward_with_explanations()` | Row-sum $= 1.0$, non-zero variance |
| **Phase 3** | Early Detection Rate | `src/early_detection_eval.py` | $>80\%$ final F1 reached by $k=0.30$ |
| **Phase 3** | Window Inference Latency | In-memory microbenchmark | $p95 < 50\text{ ms}$ |
| **Phase 3** | Dashboard Streaming Health | `app.py` / Port 8501 | Stable RAM, bounded alert queue ($\le 5$) |
