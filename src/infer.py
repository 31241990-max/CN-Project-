
import argparse,json
from pathlib import Path
import pandas as pd,torch
from data_pipeline import FlowPreprocessor,make_windows
from model import HybridGNNTransformer
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--csv",required=True);ap.add_argument("--out-dir",default="outputs");ap.add_argument("--threshold",type=float,default=.5);a=ap.parse_args()
    out=Path(a.out_dir);meta=json.loads((out/"metadata.json").read_text());ck=torch.load(out/"best_model.pt",map_location="cpu")
    prep=FlowPreprocessor.load(out/"scaler.joblib");df=pd.read_csv(a.csv)
    ws=make_windows(df,prep,meta["window_size"],meta["stride"],meta["num_nodes"])
    model=HybridGNNTransformer(ck["node_dim"],ck["edge_dim"],max_seq_len=meta["window_size"]);model.load_state_dict(ck["model_state"]);model.eval()
    rows=[]
    with torch.no_grad():
        for w in ws:
            p=torch.softmax(model(torch.from_numpy(w.node_features),torch.from_numpy(w.edge_index).long(),torch.from_numpy(w.edge_features)),1)[:,1].numpy()
            rows += list(zip(w.timestamps,p,(p>=a.threshold).astype(int)))
    pd.DataFrame(rows,columns=["FLOW_START_TIMESTAMP","attack_probability","prediction"]).drop_duplicates("FLOW_START_TIMESTAMP").to_csv(out/"predictions.csv",index=False)
    print("Saved",out/"predictions.csv")
if __name__=="__main__":main()
