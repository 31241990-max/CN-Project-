import pandas as pd

from src.data_pipeline import FlowPreprocessor, make_pyg_dataloader, make_pyg_windows


def test_make_pyg_windows_creates_graph_data_objects():
    df = pd.DataFrame(
        {
            "FLOW_START_TIMESTAMP": [1.0, 2.0, 3.0, 4.0, 5.0],
            "src_node_id": [0, 1, 0, 1, 0],
            "dst_node_id": [1, 2, 2, 3, 3],
            "src_port": [80, 443, 80, 443, 80],
            "dst_port": [443, 80, 443, 80, 443],
            "protocol": [6, 6, 6, 6, 6],
            "duration": [0.1, 0.2, 0.1, 0.2, 0.1],
            "IN_BYTES": [10.0, 20.0, 15.0, 25.0, 12.0],
            "OUT_BYTES": [5.0, 9.0, 7.0, 8.0, 6.0],
            "IN_PKTS": [1.0, 2.0, 1.0, 2.0, 1.0],
            "OUT_PKTS": [1.0, 1.0, 1.0, 1.0, 1.0],
            "FLOW_IAT_MEAN": [0.01, 0.02, 0.01, 0.02, 0.01],
            "FLOW_IAT_STD": [0.005, 0.01, 0.005, 0.01, 0.005],
            "TCP_FLAGS": [16, 18, 16, 18, 16],
            "label": [0, 1, 0, 1, 0],
        }
    )

    prep = FlowPreprocessor().fit(df)
    graphs = make_pyg_windows(df, prep, window_size=3, stride=2, num_nodes=4)

    assert len(graphs) == 2
    sample = graphs[0]
    assert hasattr(sample, "x")
    assert hasattr(sample, "edge_index")
    assert hasattr(sample, "edge_attr")
    assert sample.edge_index.shape[0] == 2
    assert sample.edge_attr.shape[1] == len(sample.edge_attr[0])

    loader = make_pyg_dataloader(df, prep, window_size=3, stride=2, batch_size=1, num_nodes=4)
    batch = next(iter(loader))
    assert hasattr(batch, "batch")
    assert batch.x.dim() == 2
    assert batch.edge_attr.dim() == 2
