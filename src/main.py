import torch 

import torch_geometric.nn as pyg
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool

class GraphTransformer(nn.Module):
    def __init__(self,in_channels=320):
        super().__init__()


        self.TransformerConv_1=pyg.GPSConv(
                              channels=in_channels,
                              conv=pyg.SAGEConv(in_channels,in_channels),
                              heads=4,
                              dropout=0.1,
                              act="RELU",
                              norm="layer_norm")
        
        self.TransformerConv_2=pyg.GPSConv(
                              channels=in_channels,
                              conv=pyg.SAGEConv(in_channels,in_channels),
                              heads=4,
                              dropout=0.1,
                              act="RELU",
                              norm="layer_norm")
        self.norm=nn.LayerNorm(in_channels)
        self.out=nn.Sequential(nn.Dropout(0.118),nn.Linear(in_channels,in_channels))
        self.classifier=nn.Linear(in_channels*2,4)

    def forward(self,edge_index,x,batch,vit_emb):
        
        x=self.TransformerConv_1(x,edge_index)
        x=self.TransformerConv_2(x,edge_index)
        x=self.norm(x)
        x=global_mean_pool(x,batch)
        
        x=self.out(x)
        x=self.classifier(vit_emb@x)
        return x
    

import os
import torch
import torch.nn as nn
import pandas as pd

from torch_geometric.data import Data,Dataset
from torch_geometric.loader import DataLoader

###
DATA_ROOT=r"D:\Python311\Pets\GraphT\data"
###

class GraphTransf(Dataset):
    def __init__(self,split_dir):
        super().__init__()
        tensor_dir=os.path.join(split_dir,"Tensors")
        csv_file=os.path.join(split_dir,f"{os.path.basename(split_dir)}.csv")
        df=pd.read_csv(csv_file)

        label_map={
            "alpha":0,
            "beta":1,
            "alpha_beta":2,
            "fewss":3
            }
        
        self.samples=[]
        for _,row in df.iterrows():
            pdb=row["PDB"]
            label=label_map[row["CLASS"]]
            path=os.path.join(tensor_dir,f"{pdb}.pt")
            if os.path.exists(path):
                self.samples.append((path,label))

    def len(self):
        return len(self.samples)
    
    def get(self,idx):
        path,label=self.samples[idx]
        data=torch.load(path,map_location="cpu")

        edge_index=data["G_edge_index"].long()
        node_attr=data["esm_residue_emb"].float()
        vit_emb=data["vit_struct_emb"]
        data=Data(edge_index=edge_index,x=node_attr,y=torch.tensor(label,dtype=torch.long))
        data.vit_emb=vit_emb
        return data


trainSplit=DataLoader(
    GraphTransf(os.path.join(DATA_ROOT, "train")),
    batch_size=1,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True)

valSplit=DataLoader(
    GraphTransf(os.path.join(DATA_ROOT, "val")),
    batch_size=1,
    shuffle=True)

testSplit=DataLoader(
    GraphTransf(os.path.join(DATA_ROOT, "test")),
    batch_size=1,
    shuffle=True)


import torch 
import torch.nn as nn

###
DEVICE=torch.device("cuda")
EPOCHS=32
###

def main():
    model=GraphTransformer().to(DEVICE)

    loss_fn=nn.CrossEntropyLoss()
    lr=5e-3
    optimizer=torch.optim.AdamW( #type:ignore
        model.parameters(),
        lr=1.5e-4,
        weight_decay=1e-4)
    
    for i in range(EPOCHS):

        model.train()
        optimizer.zero_grad()

        for tensors in trainSplit:
            
            tensors=tensors.to(DEVICE)
            g_node_attr=tensors.x
            g_edge_index=tensors.edge_index
            v_embedding=tensors.vit_struct_emb

            logits=model(g_edge_index,g_node_attr,tensors.batch,v_embedding)
            loss=loss_fn(logits,tensors.y)
            loss.backward()
            optimizer.step()

            


            
            
        