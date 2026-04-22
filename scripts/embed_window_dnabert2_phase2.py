import pandas as pd
import numpy as np
from Bio import SeqIO

PURE_SUBTYPES = ["A","B","C","D","F1","F2","G","H","J","K","L"]
##model work with numbers
ST2ID ={}
for i, s in enumerate(PURE_SUBTYPES):
    ST2ID[s]=i

ID2ST = {}
for s, i in ST2ID.items():
    ID2ST[i] = s

#clean input sequences
def ungap(seq):
    seq = str(seq)
    seq = seq.replace("-", "")
    seq = seq.replace(".", "")
    seq = seq.upper()
    return seq

#make window
def make_windows(seq, win, stride):
    L = len(seq)
    for s in range(0, L-win+1, stride):
        w = seq[s:s+win]
        yield s, w

#where to label the window
def get_anchor(start, win, label_mode):
    if label_mode == "center":
        return start+win//2
    else:
        return start

#load the labels
def load_labels(labels_csv:str):
    df = pd.read_csv(labels_csv).set_index("ID")
    if "BP2" in df.columns and "Seg3" in df.columns:
        mode = "3seg"
    else:
        mode = "2seg"
    return df, mode

#label windows
def window_label(row, start, win, label_mode):
    anchor = get_anchor(start, win, label_mode)

    if "BP2" in row.index and "Seg3" in row.index:
        bp1 = int(row["BP1"])
        bp2 = int(row["BP2"])

        if anchor < bp1:
            return row["Seg1"]
        elif anchor < bp2:
            return row["Seg2"]
        else:
            return row["Seg3"]
        
    else:
        bp = int(row["BP"])

        if anchor < bp:
            return row["Seg1"]
        else:
            return row["Seg2"]

#embedding from https://huggingface.co/quietflamingo/dnabert2-no-flashattention
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig


def embed_one_sequence(seq, tokenizer, model, device):
    inputs = tokenizer(seq, return_tensors="pt").to(device)
    outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state
    #last_hidden = model(inputs)[0] ##should be same
    emb = last_hidden.mean(dim=1)

    #no training needed, so detached without gradients and move tensors from 
    #gpu to cpu numpy arrays
    return emb.detach().cpu().numpy()

#GPU computation goal to imput a list of sequence windows rather than one by one
def forward_batch(texts, tokenizer, model, device, win, max_nt):
    inputs = tokenizer(
        texts,
        return_tensor="pt",
        padding=True,
        truncation=True,
        max_length=min(win, max_nt)
    ).to(device)

    outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state
    emb = last_hidden.mean(dim=1)

    return emb.detach().cpu().numpy()


device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    "quietflamingo/dnabert2-no-flashattention",
    trust_remote_code=True,
)

config = BertConfig.from_pretrained(
    "quietflamingo/dnabert2-no-flashattention",
)

model = AutoModel.from_pretrained(
    "quietflamingo/dnabert2-no-flashattention",
    config=config
)

model.eval()
model = model.to(device)

batch_size = 16
win = 1024
stride = 32
max_nt = 2048
label_mode = "start"
fasta = "./synthetic.fasta"

seq_counter = 0
for rec in SeqIO.parse(fasta, "fasta"):
    sid = rec.id
    seq = ungap(rec.seq)
    row =None

    if len(seq) < win: 
        continue
    seq_idx = seq_counter
    seq_counter += 1 

    batch_texts = []
    batch_starts = []
    batch_labels =[] 

    X_buf = [] 
    seq_idx_buf = [] 
    start_buf = [] 
    y_buf = [] 

    for start, wseq in make_windows(seq, win, stride):
        batch_texts.append(wseq)
        batch_starts.append(start)
        if row is not None:
            lab = window_label(row, start, win, label_mode):
            batch_labels.append(ST2ID[lab])
            
        if len(batch_texts) == batch_size:
            emb = forward_batch(batch_texts, tokenizer, model, device, win, max_nt)

            for i in range(emb.shape[0]):
                X_buf.append(emb[i])
                seq_idx_buf.append(emb[seq_idx])
                start_buf.append(batch_starts[i])

                if row is not None:
                    y_buf.append(batch_labels[i])
            batch_texts = []
            batch_starts = []
            batch_labels = []




