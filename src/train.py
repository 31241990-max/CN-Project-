
import argparse,json,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,average_precision_score,f1_score,precision_recall_curve,precision_score,recall_score,roc_auc_score
from data_pipeline import FlowPreprocessor,make_prefix_windows,make_windows,save_metadata
from model import HybridGNNTransformer,FocalLoss

def seed_everything(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def calc_metrics(logits,y,threshold=0.5):
    p=torch.softmax(logits,1)[:,1].cpu().numpy()
    yy=y.cpu().numpy(); pred=(p>=threshold).astype(int)
    m={"accuracy":accuracy_score(yy,pred),
       "precision":precision_score(yy,pred,zero_division=0),
       "recall":recall_score(yy,pred,zero_division=0),
       "f1":f1_score(yy,pred,zero_division=0)}
    has_both_classes=len(np.unique(yy))==2
    m["roc_auc"]=roc_auc_score(yy,p) if has_both_classes else None
    m["pr_auc"]=average_precision_score(yy,p) if has_both_classes else None
    if has_both_classes:
        precision,recall,thresholds=precision_recall_curve(yy,p)
        f1_values=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
        m["best_f1_threshold"]=float(thresholds[int(np.argmax(f1_values))]) if len(thresholds) else float(threshold)
    else:
        m["best_f1_threshold"]=float(threshold)
    m["threshold"]=float(threshold)
    return m

def epoch(model,windows,optimizer,criterion,device,training,threshold=0.5,clip=1.0):
    model.train(training); losses=[]; ys=[]; logits_all=[]
    if training: optimizer.zero_grad(set_to_none=True)
    for i,w in enumerate(windows):
        x=torch.from_numpy(w.node_features).to(device)
        ei=torch.from_numpy(w.edge_index).long().to(device)
        ef=torch.from_numpy(w.edge_features).to(device)
        y=torch.from_numpy(w.labels).long().to(device)
        with torch.set_grad_enabled(training):
            logits=model(x,ei,ef); loss=criterion(logits,y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()));ys.append(y.detach().cpu());logits_all.append(logits.detach().cpu())
    m=calc_metrics(torch.cat(logits_all),torch.cat(ys),threshold);m["loss"]=float(np.mean(losses))
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",default="processed");ap.add_argument("--out-dir",default="outputs")
    ap.add_argument("--window-size",type=int,default=128);ap.add_argument("--stride",type=int,default=64)
    ap.add_argument("--epochs",type=int,default=10);ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--dropout",type=float,default=.2);ap.add_argument("--weight-decay",type=float,default=1e-4)
    ap.add_argument("--patience",type=int,default=3);ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--class-weight-mode",choices=("balanced","none"),default="balanced")
    ap.add_argument("--loss",choices=("focal","cross-entropy"),default="focal")
    args=ap.parse_args();seed_everything(args.seed)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    tr=pd.read_csv(Path(args.data_dir)/"train.csv");va=pd.read_csv(Path(args.data_dir)/"val.csv");te=pd.read_csv(Path(args.data_dir)/"test.csv")
    prep=FlowPreprocessor().fit(tr);prep.save(out/"scaler.joblib")
    n=int(max(tr.src_node_id.max(),tr.dst_node_id.max(),va.src_node_id.max(),va.dst_node_id.max(),te.src_node_id.max(),te.dst_node_id.max())+1)
    tw=make_prefix_windows(tr,prep,args.window_size,args.stride,n)
    vw=make_prefix_windows(va,prep,args.window_size,args.stride,n)
    ew=make_windows(te,prep,args.window_size,args.stride,n)
    edge_dim=len(prep.scaler.feature_names_in_)+1;node_dim=len(prep.scaler.feature_names_in_)+1
    save_metadata(out/"metadata.json",n,edge_dim,args.window_size,args.stride)
    counts=np.bincount(tr.label.astype(int),minlength=2)
    weights=counts.sum()/(2*np.maximum(counts,1)) if args.class_weight_mode=="balanced" else None
    print("Train class counts:",counts.tolist())
    model=HybridGNNTransformer(node_dim,edge_dim,dropout=args.dropout,max_seq_len=args.window_size).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=1)
    alpha=torch.tensor(weights,dtype=torch.float32,device=device) if weights is not None else None
    loss_fn=FocalLoss(alpha,gamma=2) if args.loss=="focal" else torch.nn.CrossEntropyLoss(weight=alpha)
    best=-float("inf");stale=0;history=[];best_threshold=.5
    for ep in range(1,args.epochs+1):
        trm=epoch(model,tw,opt,loss_fn,device,True)
        with torch.no_grad(): vam=epoch(model,vw,opt,loss_fn,device,False)
        validation_score=vam["pr_auc"] if vam["pr_auc"] is not None else vam["f1"]
        scheduler.step(validation_score)
        history.append({"epoch":ep,"lr":opt.param_groups[0]["lr"],"train":trm,"val":vam})
        print(f"Epoch {ep:02d} | train F1@0.5 {trm['f1']:.4f} | val PR-AUC {vam['pr_auc']:.4f} | val F1@0.5 {vam['f1']:.4f} | val F1 threshold {vam['best_f1_threshold']:.4f}")
        if validation_score>best:
            best=validation_score;stale=0;best_threshold=vam["best_f1_threshold"]
            torch.save({"model_state":model.state_dict(),"config":vars(args),"node_dim":node_dim,"edge_dim":edge_dim,"num_nodes":n,"decision_threshold":best_threshold,"validation_pr_auc":validation_score},out/"best_model.pt")
        else:
            stale+=1
            if stale>=args.patience: break
    ckpt=torch.load(out/"best_model.pt",map_location=device);model.load_state_dict(ckpt["model_state"])
    best_threshold=float(ckpt.get("decision_threshold",best_threshold))
    save_metadata(out/"metadata.json",n,edge_dim,args.window_size,args.stride,best_threshold)
    with torch.no_grad(): test=epoch(model,ew,opt,loss_fn,device,False,best_threshold)
    (out/"history.json").write_text(json.dumps(history,indent=2))
    (out/"test_metrics.json").write_text(json.dumps(test,indent=2))
    print("TEST:",json.dumps(test,indent=2))
if __name__=="__main__":main()
