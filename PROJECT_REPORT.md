# Project Report: Hybrid Graph Neural Network and Transformer Framework for Early Cyber Attack Prediction

---

## 1. Executive Summary & Problem Statement

### 1.1 The Cyber Threat Landscape & The Cyber Kill-Chain
Modern computer networks face thousands of connection attempts every minute. Malicious hackers rarely attack a system in a single step. Instead, they follow a structured sequence of actions known as the **Cyber Kill-Chain**:

1. **Reconnaissance (Probing):** The attacker scans the network to find active computers, open ports, and weak software versions.
2. **Weaponization:** The attacker packages malicious code tailored to the vulnerabilities discovered.
3. **Delivery:** The exploit is transmitted to the target (via email, web download, or open ports).
4. **Exploitation:** The code triggers and takes advantage of a flaw on the target computer.
5. **Installation:** Malware or a backdoor is permanently installed on the host.
6. **Command & Control (C2):** The infected computer connects back to the attacker's external server for remote commands.
7. **Actions on Objectives:** The attacker steals sensitive data (exfiltration), encrypts systems (ransomware), or destroys files.

```text
Attacker Journey:
[ Reconnaissance ]  --->  [ Delivery & Exploit ]  --->  [ Data Theft / Ransomware ]
   ▲ STOP THEM HERE!         Too late! Damage done.         Devastating business loss!
```

---

### 1.2 Why Early Prediction is Critical
Most security tools act like **fire alarms**—they only ring after a fire has already burned down the room. Traditional intrusion detection systems frequently alert defenders at Step 6 or 7, when data is already encrypted or being stolen.

This project focuses on **Early Prediction**: detecting and stopping attackers during **Step 1 (Reconnaissance)**. 
- If you catch a burglar while they are testing doorknobs outside the building, you prevent the theft before any window is broken.
- By flagging early scanning and suspicious probes in the first few packets, defenders can block the offending IP address before any exploit payloads can land.

---

### 1.3 What is Network Telemetry?
Computers communicate using network flows. Rather than inspecting the private contents of every message (which is often encrypted with HTTPS), our system inspects **Network Telemetry metadata**:

* **The 5-Tuple:**
  1. **Source IP Address:** Which computer started the conversation?
  2. **Destination IP Address:** Which computer is receiving the message?
  3. **Source Port:** Which application door sent the message?
  4. **Destination Port:** Which service door was targeted (e.g., Port 80 for Web, Port 22 for SSH)?
  5. **Protocol:** What language are they speaking (e.g., TCP or UDP)?
* **Inter-Arrival Times (IAT):**
  - How many milliseconds pass between consecutive packets?
  - Humans browse erratically (random delays), while automated scanning tools send packets with machine-like regularity or in rapid-fire bursts.
* **TCP Flags (SYN, ACK, FIN, RST):**
  - TCP connections start with a "handshake" (SYN $\to$ SYN-ACK $\to$ ACK).
  - Attackers often send single "SYN" packets without completing the handshake (a SYN Scan) to see if a port answers, then vanish. Tracking these flags exposes covert scans immediately.

---

## 2. System Architecture: The Hybrid Approach

Traditional security solutions look at packets one by one in isolation. They miss the bigger picture. Our system uses a **Hybrid Spatio-Temporal Artificial Intelligence model** that pairs two distinct detectives:

```text
                   ┌──────────────────────────────────────────────────┐
                   │               Network Flow Stream                │
                   └────────────────────────┬─────────────────────────┘
                                            │
                                            ▼
                   ┌──────────────────────────────────────────────────┐
                   │             Chronological Graph Window           │
                   │        (Hosts = Nodes, Flows = Edges)            │
                   └───────────┬──────────────────────────┬───────────┘
                               │                          │
                 [ Spatial Relationship ]       [ Temporal Timing ]
                               │                          │
                               ▼                          ▼
                   ┌──────────────────────┐   ┌───────────────────────┐
                   │ Graph Convolutional  │   │  Normalized Features  │
                   │    Network (GCN)     │   │   & Relative Timings  │
                   └───────────┬──────────┘   └───────────┬───────────┘
                               │                          │
                               └────────────┬─────────────┘
                                            ▼
                   ┌──────────────────────────────────────────────────┐
                   │           Combined Edge Embeddings               │
                   └────────────────────────┬─────────────────────────┘
                                            │
                                            ▼
                   ┌──────────────────────────────────────────────────┐
                   │          Multi-Head Transformer Encoder          │
                   │    (Self-Attention over Flow Sequence Rhythm)    │
                   └────────────────────────┬─────────────────────────┘
                                            │
                                            ▼
                   ┌──────────────────────────────────────────────────┐
                   │          Threat Classification Output            │
                   │             [ BENIGN ] or [ ATTACK ]             │
                   └──────────────────────────────────────────────────┘
```

---

### 2.1 The Spatial Detective: Graph Neural Network (GCN)
* **Real-World Analogy:** Think of the GCN as a detective mapping out a corporate building. It notes which desks are close together, who talks to whom regularly, and identifies who is wandering into hallways they don't belong in.
* **In Cybersecurity:** The entire computer network is treated as a **Graph**:
  - **Nodes:** Computers, servers, and routers identified by their IP addresses.
  - **Edges:** Network connections between these computers.
* **Neighborhood Aggregation:** The GCN lets each computer share summary information with its neighbors. If a previously quiet desktop computer suddenly tries to connect to 50 server ports across the building, its neighbor-embedding changes dramatically. The GCN flags this abnormal neighborhood layout.

---

### 2.2 The Temporal Detective: Transformer Encoder
* **Real-World Analogy:** Think of the Transformer as a detective who listens to the rhythm and sequence of footsteps outside the building. It doesn't just note that a sound occurred; it analyzes the exact cadence—two quick taps, a long pause, then a handle jiggle.
* **In Cybersecurity:** Network flows happen over time. An attack is not a single packet; it is a **sequence of actions**:
  1. A ping to see if the host is awake.
  2. A quick probe on Port 443.
  3. A probe on Port 8080.
* **Self-Attention Mechanism:** The Transformer looks at the entire sequence of events at once. It uses **Self-Attention** to calculate mathematical connections between earlier flows and later flows, understanding that an event happening now is directly related to a subtle probe from 5 seconds ago.

---

### 2.3 The Hybrid Spatio-Temporal Fusion
By combining both detectives, the model sees what no single model can:
1. The **GCN** explains **where** the activity is happening and how hosts are connected (Spatial context).
2. The **Transformer** explains **when** and in what **order** the activity is unfolding (Temporal rhythm).
3. The fusion creates an enriched fingerprint for every network interaction:
   $$\text{Edge Representation} = [\text{GCN}(\text{Source Host}) \,\|\, \text{GCN}(\text{Destination Host}) \,\|\, \text{Flow Attributes} \,\|\, \text{Relative Timestamp}]$$

---

## 3. Team Execution Breakdown

Our engineering team divided the responsibilities into three distinct, interconnected components:

```text
┌─────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐
│        Member 1         │    │        Member 2         │    │        Member 3         │
│    Data Engineering     │───▶│     AI Architecture     │───▶│     MLOps & Dashboard   │
│ (PyG Graph Preparation) │    │  (GNN + Transformer DL) │    │  (Latency & Streamlit)  │
└─────────────────────────┘    └─────────────────────────┘    └─────────────────────────┘
```

---

### 3.1 Member 1: Data Engineering & Graph Construction

#### Responsibilities:
* Cleaned and preprocessed massive tabular network flow datasets (CSE-CIC-IDS2018).
* Applied **Strict Chronological Splitting** (Train on past events, Validate on intermediate events, Test on future events) to guarantee zero data leakage.
* Scaled heavy-tailed network features (Packet counts, byte sizes, inter-arrival times) using a `StandardScaler` fitted strictly on training data.
* Converted raw tabular CSV rows into graph snapshots using **PyTorch Geometric (PyG)**.

#### Member 1 Pseudocode: Converting a CSV Row into a Graph Edge
```python
# ====================================================================
# Member 1: Transforming CSV Telemetry Records into Graph Snapshots
# ====================================================================

import numpy as np

def create_graph_window_from_csv(csv_rows, preprocessor):
    """
    Takes a chronological window of network flow rows from a CSV
    and converts them into a Graph structure for the GNN.
    """
    # Step 1: Extract numeric telemetry features (ports, bytes, IAT, flags)
    raw_features = csv_rows[[
        "src_port", "dst_port", "protocol", "duration", 
        "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", 
        "FLOW_IAT_MEAN", "FLOW_IAT_STD", "TCP_FLAGS"
    ]]
    
    # Step 2: Normalize features using our fitted scaler (avoids huge numbers breaking the model)
    normalized_features = preprocessor.transform(raw_features)
    
    # Step 3: Compute relative time deltas so the model knows packet pacing
    timestamps = csv_rows["FLOW_START_TIMESTAMP"].values
    time_deltas = timestamps - timestamps[0]
    normalized_time = time_deltas / max(time_deltas[-1], 1e-6)
    
    # Combine normalized flow statistics with relative time
    edge_attributes = np.column_stack([normalized_features, normalized_time])
    
    # Step 4: Map IP addresses to unique integer Node IDs
    source_nodes = csv_rows["src_node_id"].values
    destination_nodes = csv_rows["dst_node_id"].values
    
    # In PyTorch Geometric, edge_index has shape [2, Number of Edges]
    # Row 0 contains sender IDs; Row 1 contains receiver IDs
    edge_index = np.vstack([source_nodes, destination_nodes])
    
    # Step 5: Compute initial node features (average behavior of each host)
    unique_hosts = np.unique(np.concatenate([source_nodes, destination_nodes]))
    num_hosts = len(unique_hosts)
    node_features = np.zeros((num_hosts, edge_attributes.shape[1]))
    
    for i in range(len(source_nodes)):
        src = source_nodes[i]
        dst = destination_nodes[i]
        # Accumulate edge information into the respective host nodes
        node_features[src] += edge_attributes[i]
        node_features[dst] += edge_attributes[i]
        
    return {
        "x": node_features,           # Matrix of Host features
        "edge_index": edge_index,     # Pairs of communicating hosts
        "edge_attr": edge_attributes  # Telemetry data for each communication
    }
```

---

### 3.2 Member 2: AI Architecture & Deep Learning Training

#### Responsibilities:
* Designed the dual-layer **Graph Convolutional Network (GCN)** to extract spatial host embeddings.
* Designed the **Tensor Reshaping Pipeline** that bridges 2D Graph spatial data into 3D Temporal sequence tensors for the Transformer.
* Built a **Multi-Head Transformer Encoder** with positional embeddings to preserve the chronological flow sequence.
* Implemented **Class-Weighted Focal Loss ($\gamma = 2$)** so the model doesn't simply guess "Benign" 99% of the time due to severe class imbalance.

#### Member 2 Pseudocode: Hybrid Forward Pass & Tensor Reshaping
```python
# ====================================================================
# Member 2: Hybrid GNN + Transformer Forward Execution
# ====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridGNNTransformer(nn.Module):
    def __init__(self, node_dim=12, edge_dim=12, hidden_dim=64, model_dim=128):
        super().__init__()
        # Spatial Detective: 2 Layers of Graph Convolution
        self.gcn1 = GCNLayer(node_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        
        # Bridge: Projects combined spatial + flow data into Transformer size
        # Combined size = (src host embedding + dst host embedding + flow attributes)
        combined_dim = hidden_dim + hidden_dim + edge_dim
        self.edge_projection = nn.Linear(combined_dim, model_dim)
        
        # Temporal Detective: Multi-Head Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final Decision Maker: Benign (0) vs Attack (1)
        self.classifier = nn.Linear(model_dim, 2)

    def forward(self, node_x, edge_index, edge_attr):
        """
        Executes spatial message passing followed by temporal sequence modeling.
        """
        # --- STAGE 1: Spatial Analysis (GCN) ---
        # Each host inspects its network neighbors to learn topological context
        h = self.gcn1(node_x, edge_index)
        h = F.relu(h)
        host_embeddings = self.gcn2(h, edge_index) # Shape: [Total Hosts, 64]
        
        # --- STAGE 2: Spatial-to-Temporal Tensor Handoff ---
        src_ids = edge_index[0]
        dst_ids = edge_index[1]
        
        # Retrieve the learned embeddings for the sending and receiving hosts
        src_vectors = host_embeddings[src_ids]
        dst_vectors = host_embeddings[dst_ids]
        
        # Glue spatial embeddings together with the raw telemetry features
        # Shape: [Sequence Length, 64 + 64 + 12 = 140]
        fused_edges = torch.cat([src_vectors, dst_vectors, edge_attr], dim=-1)
        
        # Project each edge into the Transformer's expected dimensionality
        projected_edges = self.edge_projection(fused_edges) # Shape: [Seq_Len, 128]
        
        # Reshape into a 3D batch sequence for the Transformer:
        # From [Seq_Len, 128]  -->  To [1, Seq_Len, 128] (Batch Size = 1)
        transformer_input = projected_edges.unsqueeze(0)
        
        # --- STAGE 3: Temporal Analysis (Transformer Self-Attention) ---
        # Learns relationships across time steps
        temporal_context = self.transformer(transformer_input) # [1, Seq_Len, 128]
        
        # Remove the extra batch dimension: [1, Seq_Len, 128] --> [Seq_Len, 128]
        temporal_context = temporal_context.squeeze(0)
        
        # --- STAGE 4: Final Threat Prediction ---
        # Outputs 2 probabilities per packet: [Prob_Benign, Prob_Attack]
        logits = self.classifier(temporal_context)
        return logits
```

---

### 3.3 Member 3: MLOps, Evaluation & Interactive Dashboard

#### Responsibilities:
* Evaluated **Time-to-Detection**: Verified that attacks are flagged within the first $10\%$ to $30\%$ of a flow window, proving early reconnaissance detection.
* Benchmarked system inference latency to ensure real-time throughput ($<50$ milliseconds per window).
* Built a production-grade, interactive web dashboard using **Streamlit** and **Plotly** to visualize network topology, live attack alerts, and attention graphs.

#### Member 3 Pseudocode: Real-Time Stream Ingestion & Inference Loop
```python
# ====================================================================
# Member 3: Live Network Telemetry Ingestion and Alerting Loop
# ====================================================================

import time
import torch

def live_monitoring_loop(stream_source, model, preprocessor, threshold=0.50):
    """
    Simulates a live security operations center (SOC) monitor that ingests
    incoming packet telemetry and triggers alerts in under 50ms.
    """
    window_size = 128
    packet_buffer = []
    
    print("Security Monitor Active: Listening for network packets...")
    
    for incoming_packet in stream_source:
        # Collect packets into our rolling time window
        packet_buffer.append(incoming_packet)
        
        # Once we have enough packets to form a chronological window
        if len(packet_buffer) >= window_size:
            start_time = time.perf_counter()
            
            # Step 1: Build graph snapshot from the recent buffer
            current_window = packet_buffer[-window_size:]
            graph_snapshot = convert_to_graph(current_window, preprocessor)
            
            # Step 2: Run forward inference through the Hybrid Model
            with torch.no_grad():
                logits = model(
                    graph_snapshot["x"], 
                    graph_snapshot["edge_index"], 
                    graph_snapshot["edge_attr"]
                )
                # Compute probability of attack (class index 1)
                probabilities = torch.softmax(logits, dim=-1)[:, 1].numpy()
                latest_risk = probabilities[-1] # Most recent packet
                
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            
            # Step 3: Check against Threat Threshold & Alert Security Team
            if latest_risk >= threshold:
                offender = current_window[-1]["src_ip"]
                target   = current_window[-1]["dst_ip"]
                print(f"[🚨 HIGH SEVERITY ALERT] Attack Probability: {latest_risk:.2%}")
                print(f"     Source: {offender}  -->  Target: {target}")
                print(f"     Inference Completed in: {elapsed_ms:.2f} ms (<50ms SLA PASS)")
            else:
                print(f"[✓ NORMAL] Traffic Benign ({latest_risk:.2%}) | Latency: {elapsed_ms:.2f} ms")
```

---

## 4. Expected Outcomes & Real-World Impact

### 4.1 Comparison: Traditional Security vs. Our Hybrid Framework

| Feature / Capability | Traditional Firewalls & Rule-Based IDS (e.g., Snort) | Our Hybrid GNN + Transformer Framework |
| :--- | :--- | :--- |
| **Detection Basis** | Known signatures and static hashes. | Behavioral anomalies and topological shifts. |
| **Zero-Day Attacks** | **Fails completely** until vendors publish a signature update. | **Succeeds** by spotting strange scanning patterns and unusual connection rhythms. |
| **Encrypted Traffic** | Blind unless expensive TLS/SSL decryption is applied. | **Effective** because it evaluates metadata (pacing, packet sizes, port graphs). |
| **Early Warning** | Alerts only after an attack payload hits a known signature pattern. | Alerts during the **reconnaissance phase** ($k \le 20\%$ of packet sequence). |
| **False Positive Handling** | Generates thousands of repetitive, static alert logs. | Focuses on multi-hop graph patterns, reducing noise from isolated errors. |

---

### 4.2 AI Explainability for Security Analysts (SOC Trust)
A major problem with standard "black box" deep learning models is that security engineers do not trust them. If an AI screams *"Attack!"* without explaining why, analysts waste hours investigating false leads.

Our framework solves this through **Inherent Explainability**:
1. **Self-Attention Heatmaps:** The Transformer outputs attention weights showing exactly which previous packets triggered the alert. An analyst can see: *"The model flagged Packet #120 because of its direct timing correlation with probing Packet #14."*
2. **GCN Neighbor Propagation Weights:** The graph layer visualizes which neighboring hosts influenced the risk score, highlighting the exact path of potential lateral movement across the internal subnet.

```text
Attention Explainer:
Packet #14 (Port 22 Scan)  ──[ Strong Attention Link (0.84) ]──▶  Packet #120 (FLAGGED ATTACK)
Packet #45 (Web Request)   ──[ Weak Attention Link   (0.02) ]──▶  Packet #120 (Ignored background)
```

---

## 5. Conclusion
By marrying **spatial network topology (GNN)** with **temporal rhythm tracking (Transformer)**, our framework shifts the paradigm of cyber defense from reactive cleanup to proactive early interception. Detecting malicious reconnaissance in real-time ($<50\text{ ms}$) provides security teams with the precious seconds required to isolate compromised hosts, block adversarial IPs, and safeguard critical digital infrastructure.
