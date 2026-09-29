
import argparse,json
from pathlib import Path
import pandas as pd,torch
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score,average_precision_score
from data_pipeline import FlowPreprocessor,make_windows
from model import HybridGNNTransformer

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data-dir",default="processed");ap.add_argument("--out-dir",default="outputs");a=ap.parse_args()
    out=Path(a.out_dir);meta=json.loads((out/"metadata.json").read_text());ck=torch.load(out/"best_model.pt",map_location="cpu")
    prep=FlowPreprocessor.load(out/"scaler.joblib");df=pd.read_csv(Path(a.data_dir)/"test.csv")
    ws=make_windows(df,prep,meta["window_size"],meta["stride"],meta["num_nodes"])
    model=HybridGNNTransformer(ck["node_dim"],ck["edge_dim"],max_seq_len=meta["window_size"]);model.load_state_dict(ck["model_state"]);model.eval()
    ys=[];ps=[]
    with torch.no_grad():
        for w in ws:
            p=torch.softmax(model(torch.from_numpy(w.node_features),torch.from_numpy(w.edge_index).long(),torch.from_numpy(w.edge_features)),1)[:,1]
            ys.append(torch.from_numpy(w.labels));ps.append(p)
    y=torch.cat(ys).numpy();p=torch.cat(ps).numpy()
    threshold=float(ck.get("decision_threshold",meta.get("decision_threshold",0.5)))
    pred=(p>=threshold).astype(int)
    report=classification_report(y,pred,target_names=["Benign","Attack"],digits=4)
    metrics={
        "threshold":threshold,
        "accuracy":float(accuracy_score(y,pred)),
        "precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),
        "f1":float(f1_score(y,pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        "pr_auc":float(average_precision_score(y,p)) if len(set(y))==2 else None,
        "confusion_matrix":confusion_matrix(y,pred).tolist(),
        "classification_report":report,
    }
    print(report);print("Confusion matrix:\n",metrics["confusion_matrix"])
    print("Threshold:",threshold,"ROC-AUC:",metrics["roc_auc"],"PR-AUC:",metrics["pr_auc"])
    (out/"classification_report.txt").write_text(report)
    (out/"test_metrics.json").write_text(json.dumps(metrics,indent=2))
if __name__=="__main__":main()
