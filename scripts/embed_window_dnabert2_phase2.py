import pandas as pd
import numpy as np

#clean input sequences
def ungap(seq):
    seq = str(seq)
    seq = seq.replace("-", "")
    seq = seq.replace(".", "")
    seq = seq.upper()
    return seq

#make window
def make_window(seq, win, stride):
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
    if "BP2" in df.columns or "Seg3" in df.columns:
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
    inputs = tokenizer(seq, return_tensors="pt"). to(device)
    outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state
    #last_hidden = model(inputs)[0] ##should be same
    emb = last_hidden.mean(dim=1)

    return emb.detach().cpu().numpy()

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    "quietflamingo/dnabert2-fix",
    trust_remote_code=True,
)

config = BertConfig.from_pretrained(
    "quietflamingo/dnabert2-fixed",
)

self.model = AutoModel.from_pretrained(
    "quietflamingo/dnabert2-fixed",
    config=config
)
