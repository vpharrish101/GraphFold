import os
import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,AutoModel


DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT=r"D:\Python311\Pets\GraphT\data"
MODEL_NAME="facebook/esm2_t6_8M_UR50D"

BATCH_SIZE=6
MAX_LEN=1024



class ESM_Dataset(Dataset):
    def __init__(self,split_dir):
        self.fasta_dir=os.path.join(split_dir,"FASTA")
        self.tensor_dir=os.path.join(split_dir,"Tensors")
        self.files=[
            f for f in os.listdir(self.fasta_dir)
            if f.endswith(".fasta")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):

        fasta_name=self.files[idx]
        pdb=fasta_name.replace(".fasta","")

        fasta_path=os.path.join(self.fasta_dir,fasta_name)
        pt_path=os.path.join(self.tensor_dir,f"{pdb}.pt")

        with open(fasta_path) as f:
            seq="".join(
                line.strip()
                for line in f
                if not line.startswith(">")
            )
        return seq,pt_path



@torch.no_grad()
def embed_batch(seqs,tokenizer,model):
    tokens=tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN)

    tokens={k:v.to(DEVICE) for k,v in tokens.items()}

    with torch.autocast("cuda",dtype=torch.float16):
        output=model(**tokens)

    hidden=output.last_hidden_state
    mask=tokens["attention_mask"]

    embs=[]

    for i in range(hidden.size(0)):
        L=int(mask[i].sum())
        embs.append(hidden[i,:L].half().cpu())

    return embs



def process_split(split,tokenizer,model):


    split_dir=os.path.join(DATA_ROOT,split)
    loader=DataLoader(
        ESM_Dataset(split_dir),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True)

    for seqs,pt_paths in loader:
        embs=embed_batch(seqs,tokenizer,model)
        for emb,pt_path in zip(embs,pt_paths):
            if not os.path.exists(pt_path): continue

            data=torch.load(pt_path,map_location="cpu")

            if "esm_residue_emb" in data: continue

            data["esm_residue_emb"]=emb
            torch.save(data,pt_path)




def main():

    tokenizer=AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=r"D:\Python311\Pets\GraphT\models")

    model=AutoModel.from_pretrained(
        MODEL_NAME,
        cache_dir=r"D:\Python311\Pets\GraphT\models").to(DEVICE)

    model.eval()
    torch.set_grad_enabled(False)

    for p in model.parameters():
        p.requires_grad=False

    for split in ["train","val","test"]:
        process_split(split,tokenizer,model)


if __name__ == "__main__":
    main()
