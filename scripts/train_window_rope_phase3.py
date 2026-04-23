
"""
Read and train embeddings from the local disk
|
Group windowns back to full length sequences
|
Feed the embeddings into Transformer
|
Per-window subtype logits
|
Model weights
|
Validation dataset - evaluate
|
Global optimize - which window to split
"""

#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data import Dataset, DataLoader

import glob
import numpy as np
from dataclasses import dataclass

PURE_SUBTYPES = ["A","B","C","D","F1","F2","G","H","J","K"]
SID2ID = {s:i for i,s in enumerate(PURE_SUBTYPES)}
ID2SID = {i:s for s, i in SID2ID.items()}

@dataclass
class Segment:
    shared_path: str
    start: int
    end: int
    
class WindowSeqDataset:
    def __init__(self, feats_dir, index_path, label_csv, seq_map, stride, boundary_k):
        self.feats_dir = feats_dir
        self.index_path = index_path
        self.label_csv = label_csv
        self.seq_map = seq_map
        self.stride = stride
        self.boundary_k = boundary_k

        
        self.n = 10 #fake dataset size for now to test
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        T = 100 #fake windown for now to test
        X = torch.randn(T, 768, dtype=torch.float32)
        y = torch.randint(0, len(PURE_SUBTYPES), (T,), dtype=torch.long)

        return {
            "X": X,
            "y": y
        }

def DataLoader(ds, batch_size, shuffle, collate_fn):

    return

def WindowTransformer(in_dim, d_model, n_layers, n_heads, n_classes, dropout, boudary_head):

    return

def collate_pad(batch):
    T_max = max(item["X"].shape[0] for item in batch) #largest number of windows per seq
    B = len(batch)

    X = torch.zeros((B, T_max, 786), dtype=torch.float32)
    y = torch.fill((B, T_max), fill_value=-100, dtype=torch.long)
    mask = torch.zeros((B, T_max), dtype=torch.bool) #convert 0 to False

    for i, item in enumerate(batch):
        t = item["X"].shape[0]
        X[i, :t] = item["X"] #for sequence i, take pos 0 -> t-1
        y[i, :t] = item["y"]
        mask[i, :t] = True

    return {
        "X": X,
        "y": y,
        "mask": mask,
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--out_model", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size_seqs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4) #learning rate for Transformer

    parser.add_arguemnt("--train_seq_map", required=True)
    parser.add_argument("--val_seq_map", required=True)
    parser.add_argument("--train_labels", default=None)
    parser.add_argument("--val_labels", default=None)
    parser.add_argument("--stride", default=64)
    parser.add_argument("--boundary_k", default=2)

    parser.add_arguement("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_boundary_head", action="store_true", default=True)
    parser.add_argument("--bw", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)

    print("Training dataset directory: ", args.train_dir)
    print("Validation dataset directory: ", args.val_dir)

    device = torch.device("cuda")

    print("Using device: ", device)

    # --- Dataset --
    train_index = os.path.join(args.train_dir, "seq_index.npz")
    val_index = os.path.join(args.val_dir, "seq_index.npz")

    train_ds = WindowSeqDataset(
        feats_dir=args.train_dir,
        index_path=train_index,
        labels=args.train_labels,
        seq_map=args.train_seq_map,
        stride=args.stride,
        boundary_k=args.boundary_k
        )
    val_ds = WindowSeqDataset(
        feats=args.train_dir,
        index=val_index,
        labels=args.val_labels,
        seq_map=args.val_seq_map,
        stride=args.stride,
        boundary_k=args.boundary_k,
    )

    # ---Dataholder ---
    train_loader = DataLoader(train_ds, batch_size=args.batch_size_seqs, shuffle=True, collate_fun=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_seqs, shuffle=False, collate_fn=collate_pad)

    print("Training sequences: ", len(train_ds))
    print("Val sequences: ", len(val_ds))
    # ---Model ---
    model = WindowTransformer(
        in_dim=768,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_class=len(PURE_SUBTYPES),
        dropout=args.dropout,
        boundary_head=args.use_boundary_head,

    ).to(device)

    # ---Optimizer ---
    # use the gradiates to optimize the model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ce = nn.CrossEntropyLoss(ignore_index=-100) #the paddings are fake, should be ignored, "-100" is a fake subtype class
    bce = nn.BCEWithLogistsLoss(reduction="none")
    best_val = -1.0 #will be updated every epoach, and the real should be in 0-1

    # --- Train loop ---

    for ep in range(1, args.epochs+1):
        print(f"Epoch: {ep}")

        model.train()
        total_loss =0
        n_batches = 0
        for batch in train_loader:
            X = batch["X"].to(device) #embedding
            y = batch["y"].to (device) #label
            mask = batch["mask"].to(device)

            opt.zero_grad() #clear old gradients

            logits, _ = model(X, mask)

            loss = ce(logits.view(-1, logits.size(-1)), y.view(-1))

            loss.backward() #compute gradiants
            opt.step() #update model weights

            total_loss += loss.item()
            n_batches += 1
        
        train_loss = total_loss/max(1, n_batches) #max here to avoid 0
        print("Train total loss is: ", train_loss)


        val_acc = 0 #placeholder
        n = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                X = batch["X"].to(device)
                y = batch["y"].to(device)
                mask = batch["mask"].to(device)

                logits, _ = model(X, mask)

                pred = logits.argmax(dim=-1)
                m = (y != -100)

                if m.sum >0:
                    acc = (pred[m] == y[m].float().mean().item())
                    val_acc += acc
                    n += 1
            val_acc = val_acc/max(1, n)
            print("Val acc: ", val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.out_model)
            print("Save the best model to: "+ args.output_model)


if __name__ == "__main()__":
    main()

