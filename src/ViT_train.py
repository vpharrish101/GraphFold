import torch
import torch.nn.functional as F
import pandas as pd
import os 

from torch.utils.data import Dataset,DataLoader

###
DATA_ROOT=r"D:\Python311\Pets\GraphT\data"
###

class ViT_Dataset(Dataset):

    def __init__(self,split_dir):
        tensor_dir=os.path.join(split_dir,"Tensors")
        csv_path=os.path.join(split_dir,f"{os.path.basename(split_dir)}.csv")
        df=pd.read_csv(csv_path)

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


    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self,idx):
        path,label=self.samples[idx]
        data=torch.load(path,map_location="cpu")
        dense=data["V_dense_map"].float()   

        dense=dense.unsqueeze(0).unsqueeze(0)
        dense=F.interpolate(
            dense,
            size=(224,224),
            mode="bilinear",
            align_corners=False)

        dense=dense.squeeze(0)     
        dense=dense.repeat(3,1,1)  
        dense=dense/20.0
        dense=dense.clamp(0,1)

        return dense, torch.tensor(label).long()
    


train_loader=DataLoader(
    ViT_Dataset(os.path.join(DATA_ROOT,"train")),
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True)

val_loader=DataLoader(
    ViT_Dataset(os.path.join(DATA_ROOT,"val")),
    batch_size=32,
    shuffle=False)

test_loader=DataLoader(
    ViT_Dataset(os.path.join(DATA_ROOT,"test")),
    batch_size=32,
    shuffle=False)

import torch
import torch.nn as nn

###
DEVICE=torch.device("cuda")
###





class ViT_Embedding(nn.Module):
    def __init__(self,ViT_model):
        super().__init__()

        self.model=ViT_model

        self.projection_head=nn.Linear(self.model.num_features,320)
        self.projection_head=nn.Sequential(
            nn.Linear(self.model.num_features,320),
            nn.LayerNorm(320))


    def forward(self,dense_map):
        x=self.model(dense_map)
        x=self.projection_head(x)

        return self.classification_head(x)

    



import os
import torch 
import torch.nn as nn
import timm

###
DEVICE=torch.device("cuda")
EPOCHS=12
###



def main():

    ViT_model=timm.create_model("vit_small_patch16_224",pretrained=True,num_classes=0)

    for p in ViT_model.parameters():
        p.requires_grad=False

    model=ViT_Embedding(ViT_model).to(DEVICE)
    loss_fn=nn.CrossEntropyLoss()

    optimizer=torch.optim.AdamW( #type:ignore
        filter(lambda p: p.requires_grad,model.parameters()),
        lr=1.5e-3,
        weight_decay=1e-2)


    for epoch in range(EPOCHS):
        model.train()

        total_loss=0
        correct=0
        total=0

        for dense_map,labels in train_loader:

            dense_map=dense_map.to(DEVICE,non_blocking=True)
            labels=labels.to(DEVICE)
            optimizer.zero_grad()
            logits=model(dense_map)
            loss=loss_fn(logits,labels)

            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            preds=logits.argmax(dim=1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)
            
        train_acc=correct/total


        model.eval()

        val_correct=0
        val_total=0

        with torch.no_grad():

            for dense_map, labels in val_loader:

                dense_map=dense_map.to(DEVICE, non_blocking=True)
                labels=labels.to(DEVICE)
                logits=model(dense_map)
                preds=logits.argmax(dim=1)

                val_correct+=(preds==labels).sum().item()
                val_total+=labels.size(0)

        val_acc=val_correct/val_total

        print(f"\nEpoch {epoch+1}")
        print("Train Loss:", total_loss/len(train_loader))
        print("Train Acc :", train_acc)
        print("Val Acc   :", val_acc)


    torch.save(model.state_dict(),"ViT_finetuned.pt")
