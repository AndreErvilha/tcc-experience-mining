"""Cluster 5: composição por rótulo, presença do pagamento e da sequência resgate -> pagamento."""
import pickle, sys
from collections import Counter
import numpy as np

root = sys.argv[1]
tf = pickle.load(open(f"{root}/data/tfidf_vectors.pkl", "rb"))
labels = np.array(pickle.load(open(f"{root}/clusters/hdbscan_tf_idf.pkl", "rb"))["cluster_labels"])
df = tf["df"]
seqs = df["text_representation"].fillna("").astype(str).str.split().tolist()
prim = np.array(tf["primary_labels"], dtype=object)
exp = np.array(tf["experiences"], dtype=object)

print("colunas:", list(df.columns))
RESG = "investimentos:view_comprovante_resgate"
PAG = "cartoes:view_comprovante_pagamento"

def seq_flag(s):
    if RESG in s and PAG in s:
        return "resgate->pagamento" if s.index(RESG) < s.index(PAG) else "pagamento->resgate"
    return "só resgate" if RESG in s else "só pagamento" if PAG in s else "nenhum"

print("\n[A] Para onde foram as sessões de cada rótulo envolvendo CDB/fatura:")
for lab in sorted(set(prim)):
    if any(k in str(lab).lower() for k in ("009", "008", "001", "resgate", "fatura com saldo")):
        idx = prim == lab
        print(f"  rótulo={lab}  N={idx.sum()}  clusters={dict(Counter(labels[idx]).most_common())}")

for c in (5, 10, 8):
    idx = np.where(labels == c)[0]
    if not len(idx):
        continue
    print(f"\n[B] Cluster {c} (N={len(idx)})")
    print("   rótulos:", dict(Counter(prim[idx]).most_common()))
    print("   experiência:", dict(Counter(exp[idx]).most_common()))
    print("   sequência:", dict(Counter(seq_flag(seqs[i]) for i in idx).most_common()))

print("\n[C] Sessões com a sequência completa resgate->pagamento, por cluster:")
full = [i for i, s in enumerate(seqs) if seq_flag(s) == "resgate->pagamento"]
print("  ", dict(Counter(labels[full]).most_common()), f"(total={len(full)})")
