#!/usr/bin/env python3

import os
import random
import argparse
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

PURE_SUBTYPES = {"A", "B","C","D","E","F1","F2","G","H","J","K","L"}

def parse_subtype(rec_id: str) ->str:
    token = rec_id.split(".", 1)[0]
    token = token.split("|", 1)[0]

    return token.strip().upper()

def get_content_range(aln: str):
    non_gap = []
    for i, ch in enumerate(aln):
        if ch != "-":
            non_gap.append(i)
    if not non_gap:
        return None, None
    return non_gap[0], non_gap[-1] 

def aln_to_real(aln: str, aln_idx: int) -> int:
    if not (0 <= aln_idx <= len(aln)):
        raise ValueError("aln_idx is out of range")
    length = len(aln[:aln_idx].replace("-",""))
    return length

def create_safe_2seg(p1, p2, min_len=500, step=10):
    s1 = str(p1.seq)
    s2 = str(p2.seq)

    if len(s1) != len(s2):
        return None
    
    s1s, s1e = get_content_range(s1)
    s2s, s2e = get_content_range(s2)

    if s1s is None or s2s is None:
        return None
    
    ov_s = max(s1s, s2s) # to choose the overlapping start pos
    ov_e = min(s1e, s2e) # to choose the overlapping end pos

    candidates = [] 
    for a in range(ov_s + 100, ov_e-100, step): #100 is the edge
        left_len = len(s1[s1s:a].replace("-", ""))
        right_len = len(s2[a:s2e].replace("-",""))

        if left_len >= min_len and right_len >= min_len:
            candidates.append(a)
    if not candidates:
        return None
    
    a = random.choice(candidates)

    chim_aln = s1[:a] + s2[a:]
    chim = chim_aln.replace("-", "")

    bp = aln_to_real(s1, a) #the aligned bp pos to real bp pos

    if not (0 <bp< len(chim)):
        return None
    
    return chim, bp, a

## updating 2 breakpoints (3 parents) - allow one parent genome being reused
def create_safe_3seg(p1,p2,p3, min_len, step=10):
    s1 = str(p1.seq)
    s2 = str(p2.seq)
    s3 = str(p3.seq)

    if len(s1) !=len(s2) or len(s1) != len(s3) or len(s2)!=len(s3):
        return None
    
    s1s, s1e = get_content_range(s1)
    s2s, s2e = get_content_range(s2)
    s3s, s3e = get_content_range(s3)

    if None in (s1s, s2s, s3s):
        return None
    
    ov_s = max(s1s, s2s, s3s)
    ov_e = min(s1e, s2e, s3e)

    if ov_e - ov_s < 500: 
        return None
    
    a1_l = [] #to store all possible a1 positions

    for a1 in range(ov_s+100, ov_e-100, step):
        if aln_to_real(s1, a1) >= min_len:
            a1_l.append(a1)
    if not a1_l:
        return None

    #shuffle the orignal list, otherwise each list starting from the same order
    random.shuffle(a1_l)

    for a1 in a1_l[:200]:
        a2_l = []
        for a2 in range(a1+min_len, ov_e-100, step):
            bp1 = aln_to_real(s1, a1)
            bp2 = bp1 + len((s2[a1:a2]).replace("-",""))
            #bp2 = len((s1[:a1] + s2[a1:a2]).replace("-","")) should be same

            seg1 = bp1
            seg2 = bp2 - bp1

            #check the seg3 length
            if seg1 < min_len or seg2 < min_len:
                continue

            seg3 = aln_to_real(s3, s3e) - aln_to_real(s3, a2)
            if seg3 < min_len:
                continue

            a2_l.append(a2)
        if not a2_l:
            continue

        a2 = random.choice(a2_l)

        chim_aln = s1[:a1]+s2[a1:a2]+s3[a2:]
        chim = chim_aln.replace("-","")

        bp1 = aln_to_real(s1,a1)
        bp2 = bp1+len((s2[a1:a2]).replace("-",""))

        if not 0 <bp1 < bp2 < len(chim):
            continue

        if (len(chim) - bp2) < min_len:
            continue

        return chim, bp1, bp2, a1, a2
    return None

        
## build pure parent pool
def build_pure_pool(aligned_fasta):
    pool = [] 

    for rec in SeqIO.parse(aligned_fasta, "fasta"):
        st = parse_subtype(rec.id)

        if st in PURE_SUBTYPES:
            pool.append(rec)
    return pool

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--aligned_fasta", required=True, help="Path to your aligned fasta file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--n_train", type=int, default=50000, help="The number of training sequences")
    parser.add_argument("--n_val", type=int, default=2000, help="Number of evaluate sequences")
    parser.add_argument("--n_test",type=int, default=2000, help="Number of testing sequences")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--min_len", type=int, default=500, help="Minimum sequence length on each side and the segments")

    parser.add_argument("--n_breakpoints", type=int, default=1, help="Number of breakpoints (1 oe 2) needed to make sythnetic genomes")

    args = parser.parse_args()

    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    pool = build_pure_pool(args.aligned_fasta)
    print(f"Pure pool size: {len(pool)}")

    def gen_split(tag, n):
        recs = [] 
        meta = [] 
        i =0

        while i < n:
            if args.n_breakpoints == 1:
                p1, p2 = random.sample(pool,2)
                
                st1 = parse_subtype(p1.id)
                st2 = parse_subtype(p2.id)

                if st1 == st2: # we do not want two same subtype
                    continue 

                out = create_safe_2seg(p1, p2, min_len=args.min_len)

                if out is None:
                    continue

                chim, bp, a = out

                sid = f"SYN2_{tag}_{i:06d}|{st1}x{st2}|BP={bp}"

                recs.append(SeqRecord(Seq(chim), id=sid, description=""))

                meta.append({
                    "ID": sid,
                    "Seg1": st1,
                    "Seg2": st2,
                    "BP": bp,
                    "Length": len(chim)
                }
                )
            
            elif args.n_breakpoints == 2:
                #to randomly produce two to three-parent genomes with 2 breakpoints
                mode = random.choice(["two", "three"]) 

                if mode == "two":
                    p1, p2 = random.sample(pool, 2)
                    p3 = p1
                else:
                    p1, p2, p3 = random.sample(pool, 3)

                st1 = parse_subtype(p1.id)
                st2 = parse_subtype(p2.id)
                st3 = parse_subtype(p3.id)

                if st1 == st2 or st2 == st3:
                    continue

                if len({st1, st2, st3}) < 2:
                    continue

                out = create_safe_3seg(p1, p2, p3, min_len=args.min_len)

                if out is None:
                    continue 

                chim, bp1, bp2, a1, a2 = out

                sid = f"SYN3_{tag}_{i:06d}|{st1}x{st2}x{st3}|BP1={bp1}|BP2={bp2}"

                recs.append(SeqRecord(Seq(chim), id=sid, description=""))

                meta.append({
                    "ID": sid,
                    "Seq1": st1,
                    "Seg2": st2,
                    "Seg3": st3,
                    "BP1": bp1,
                    "BP2": bp2,
                    "Length": len(chim)
                })
            else:
                raise ValueError("Only 1 or 2 breakpoints is supported for now")


            i += 1
        
        fasta_output = os.path.join(args.out_dir, f"{tag}.fasta")
        csv_out = os.path.join(args.out_dir, f"{tag}_label.csv")

        SeqIO.write(recs, fasta_output, "fasta")
        pd.DataFrame(meta).to_csv(csv_out, index=False)

        print(f"{fasta_output} saved")
        print(f"{csv_out} saved.")

    gen_split("train", args.n_train)
    gen_split("val", args.n_val)
    gen_split("test", args.n_test)

if __name__ == "__main__":
    main()
