# Hybrid GNN + Transformer Cyber Attack Prediction — Member 2

## Role
**Hybrid Model Architecture & Training Pipeline (AI/Deep Learning Lead)**

This implementation uses the Member 1 chronological network-flow datasets.

### 1. Spatial representation — GCN
Each communication flow is represented as a directed edge from a source host to a destination host. Node features are built from incident-flow statistics. Two Graph Convolutional Network layers aggregate local neighborhood information and learn host interaction embeddings.

### 2. Temporal representation — Transformer
For every chronological graph window, source and destination GCN embeddings are combined with the flow attributes. The resulting edge embeddings are kept in timestamp order and passed through a multi-head Transformer encoder. Self-attention captures relationships among earlier and later interactions, representing attack progression over time.

### 3. Hybrid integration
`edge_embedding = [GCN(source), GCN(destination), normalized_flow_features, relative_time]`

The Transformer converts this sequence into contextual temporal representations, followed by a binary classifier for **Benign (0)** and **Attack (1)**.

### 4. Loss design
The provided training data contains:
- Train benign: **27,907**
- Train attack: **7,093**

The implementation uses **class-weighted Focal Loss (gamma=2)** so that minority and difficult attack examples receive more learning emphasis.

### 5. Training optimization
Included:
- AdamW optimizer
- ReduceLROnPlateau learning-rate scheduler
- Dropout
- Gradient clipping
- Early stopping
- Reproducible seed
- Automatic CUDA/GPU detection
- Best-checkpoint saving
- Precision, recall, F1 and ROC-AUC evaluation
- PR-AUC and confusion matrix during evaluation

## Architecture

```text
Network Flows
     |
Chronological Window
     |
Graph Construction
  /          \
Hosts       Flows
  |           |
  +-->  GCN  <+
        GCN
         |
Source + Destination Embeddings
         |
   + Flow Features
         |
 Edge Embeddings
         |
Positional Embedding
         |
Transformer Encoder
 Multi-Head Attention
         |
Binary Classifier
    /          \
 Benign       Attack
```

## Project structure

```text
Hybrid_GNN_Transformer_Member2/
├── processed/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── src/
│   ├── data_pipeline.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── outputs/                 # created after training
├── requirements.txt
└── README.md
```

## Run on Windows / VS Code

Open this folder in VS Code.

### Install
```bash
python -m pip install -r requirements.txt
```

### Quick test
```bash
python src/train.py --epochs 2 --window-size 128 --stride 128
```

### Full training
```bash
python src/train.py --epochs 10 --window-size 128 --stride 64
```

### Evaluate
```bash
python src/evaluate.py
```

### Predict another compatible CSV
```bash
python src/infer.py --csv processed/test.csv
```

### Presentation explanation
> "My module implements the hybrid GNN-Transformer architecture. First, network traffic is represented as a graph where hosts are nodes and communication flows are edges. The GCN layers aggregate neighborhood information to learn spatial relationships between communicating hosts. We then combine the source and destination GCN embeddings with flow features and arrange these embeddings chronologically. A multi-head Transformer encoder uses self-attention to learn temporal relationships and attack progression. Because attack traffic is less frequent than benign traffic, I use class-weighted Focal Loss. The training pipeline also includes AdamW, learning-rate scheduling, dropout, gradient clipping, early stopping and GPU support."

## Important note
The train/validation/test split is chronological. The feature scaler is fitted only on the training set and then applied to validation and test data, reducing data leakage.

Accuracy should not be used alone because the classes are imbalanced. Report precision, recall, F1, ROC-AUC, PR-AUC and the confusion matrix.

# CN-Project-
