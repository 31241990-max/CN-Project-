
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self,in_dim,out_dim,dropout=0.2):
        super().__init__()
        self.linear=nn.Linear(in_dim,out_dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,edge_index):
        src,dst=edge_index[0],edge_index[1]
        n=x.size(0)
        row=torch.cat([src,dst]); col=torch.cat([dst,src])
        deg=torch.bincount(row,minlength=n).float().to(x.device)
        inv=deg.clamp_min(1).pow(-0.5)
        norm=inv[row]*inv[col]
        msg=x[row]*norm.unsqueeze(1)
        out=torch.zeros_like(x)
        out.index_add_(0,col,msg)
        return self.dropout(F.relu(self.linear(out)))

class HybridGNNTransformer(nn.Module):
    def __init__(self,node_in_dim,edge_in_dim,gnn_hidden=64,
                 edge_hidden=64,model_dim=128,nhead=4,
                 num_transformer_layers=2,ff_dim=256,dropout=0.2,
                 max_seq_len=128):
        super().__init__()
        self.gcn1=GCNLayer(node_in_dim,gnn_hidden,dropout)
        self.gcn2=GCNLayer(gnn_hidden,gnn_hidden,dropout)
        self.edge_projection=nn.Sequential(
            nn.Linear(edge_in_dim+2*gnn_hidden,edge_hidden),
            nn.ReLU(),nn.Dropout(dropout),nn.Linear(edge_hidden,model_dim))
        self.positional_embedding=nn.Parameter(torch.zeros(1,max_seq_len,model_dim))
        nn.init.normal_(self.positional_embedding,std=0.02)
        layer=nn.TransformerEncoderLayer(
            d_model=model_dim,nhead=nhead,dim_feedforward=ff_dim,
            dropout=dropout,activation="gelu",batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers=num_transformer_layers)
        self.norm=nn.LayerNorm(model_dim)
        self.classifier=nn.Linear(model_dim,2)

    def forward(self,node_x,edge_index,edge_features):
        h=self.gcn2(self.gcn1(node_x,edge_index),edge_index)
        src,dst=edge_index[0],edge_index[1]
        e=torch.cat([h[src],h[dst],edge_features],dim=-1)
        z=self.edge_projection(e).unsqueeze(0)
        z=z+self.positional_embedding[:,:z.size(1),:]
        z=self.transformer(z)
        return self.classifier(self.norm(z).squeeze(0))

    def forward_with_explanations(self,node_x,edge_index,edge_features):
        h=self.gcn2(self.gcn1(node_x,edge_index),edge_index)
        src,dst=edge_index[0],edge_index[1]
        e=torch.cat([h[src],h[dst],edge_features],dim=-1)
        z=self.edge_projection(e).unsqueeze(0)
        z=z+self.positional_embedding[:,:z.size(1),:]

        attention_weights=None
        for layer in self.transformer.layers:
            if layer.norm_first:
                normalized=layer.norm1(z)
                attended,attention_weights=layer.self_attn(
                    normalized,normalized,normalized,
                    need_weights=True,average_attn_weights=False)
                z=z+layer.dropout1(attended)
                normalized=layer.norm2(z)
                feedforward=layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(normalized)))
                )
                z=z+layer.dropout2(feedforward)
            else:
                attended,attention_weights=layer.self_attn(
                    z,z,z,need_weights=True,average_attn_weights=False)
                z=layer.norm1(z+layer.dropout1(attended))
                feedforward=layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(z)))
                )
                z=layer.norm2(z+layer.dropout2(feedforward))

        logits=self.classifier(self.norm(z).squeeze(0))
        row=torch.cat([src,dst])
        col=torch.cat([dst,src])
        degree=torch.bincount(row,minlength=node_x.size(0)).float().to(node_x.device)
        inverse_degree=degree.clamp_min(1).pow(-0.5)
        propagation_weights=inverse_degree[row]*inverse_degree[col]
        edge_count=edge_index.size(1)
        propagation_weights=torch.stack(
            [propagation_weights[:edge_count],propagation_weights[edge_count:]],dim=0
        )
        return logits, {
            "attention": attention_weights.squeeze(0),
            "gcn_edge_weights": propagation_weights,
        }

class FocalLoss(nn.Module):
    def __init__(self,alpha=None,gamma=2.0):
        super().__init__()
        self.gamma=gamma
        self.register_buffer("alpha",alpha if alpha is not None else torch.tensor([]))
    def forward(self,logits,targets):
        ce=F.cross_entropy(logits,targets,
            weight=self.alpha if self.alpha.numel() else None,reduction="none")
        pt=torch.exp(-ce)
        return (((1-pt)**self.gamma)*ce).mean()
