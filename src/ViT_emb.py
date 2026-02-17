import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ViT_Dataset(Dataset):

    def __init__(self,pt_dir):
        self.files=[
            os.path.join(pt_dir,f)
            for f in os.listdir(pt_dir)
            if f.endswith(".pt")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):

        path=self.files[idx]
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

        dense=dense.clamp(max=20.0)/20.0

        return dense,path


import torch
import torch.nn as nn
import timm


class ViT_Embedding(nn.Module):

    def __init__(self,backbone="vit_small_patch16_224",embed_dim=320):
        super().__init__()

        self.vit=timm.create_model(
            backbone,
            pretrained=False, 
            num_classes=0)

        self.projection_head=nn.Sequential(
            nn.Linear(self.vit.num_features,embed_dim),
            nn.LayerNorm(embed_dim))

    def forward(self,x):
        feats=self.vit(x)
        return self.projection_head(feats)



from torch.utils.data import DataLoader

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT=r"D:\Python311\Pets\GraphT\data"
CHECKPOINT="ViT_finetuned.pt"


def generate_embeddings(split):

    pt_dir=os.path.join(DATA_ROOT,split,"Tensors")
    dataset=ViT_Dataset(pt_dir)
    loader=DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    model=ViT_Embedding()
    checkpoint=torch.load(CHECKPOINT,map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()


    with torch.no_grad():
        for imgs,paths in loader:

            imgs=imgs.to(DEVICE,non_blocking=True)
            with torch.autocast("cuda",dtype=torch.float16):
                embeddings=model(imgs)

            embeddings=embeddings.cpu()

            for emb,path in zip(embeddings,paths):
                data=torch.load(path,map_location="cpu")
                data["vit_struct_emb"]=emb.half()
                torch.save(data,path)


if __name__ == "__main__":

    for split in ["train","val","test"]:
        generate_embeddings(split)
