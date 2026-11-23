"""Sessões 'resgatou E pagou' (RF): os desvios (open finance, PIX, ...) afastam essas sessões entre si
mais nos embeddings (LLM) do que no TF-IDF? E os embeddings veem a sessão inteira (truncamento)?

1) Desvios: produtos visitados em cada sessão RF, além de investimentos/cartões.
2) Truncamento: quantos eventos cabem na janela do MiniLM (256 subtokens, tokenizer real) e estimativa
   para o Gemini (2048 tokens, ~4 caracteres/token). O resgate e o pagamento estão dentro da janela?
3) Sensibilidade: entre pares de sessões RF, correlação (Spearman) entre a distância em cada espaço e
   a) diferença nos produtos visitados (Jaccard),  b) diferença nos eventos (Jaccard, sessão inteira),
   c) diferença nos eventos só do começo da sessão (o que o MiniLM vê),  d) diferença de tamanho.
   Distâncias convertidas em percentil sobre todos os pares de sessões, para comparar espaços.

Uso (da raiz): python analises/desvios_e_truncamento.py .
Requer: tokenizers, huggingface_hub (baixa só o tokenizer público do all-MiniLM-L6-v2).
"""
import pickle, sys
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import pairwise_distances

root = sys.argv[1] if len(sys.argv) > 1 else "."
tf = pickle.load(open(f"{root}/data/tfidf_vectors.pkl", "rb"))
df = tf["df"]
seqs = df["text_representation"].fillna("").astype(str).str.split().tolist()
resg = df["is_cdb_resgatado"].values == 1
pag = df["is_fatura_paga_saldo"].values == 1
rf = np.where(resg & pag)[0]
print(f"Sessões RF (resgatou e pagou): {len(rf)}")

prod = lambda s: {e.split(":")[0] for e in s}
DESVIO = lambda s: sorted(prod(s) - {"investimentos", "cartoes", "canais"})

# 1) desvios
print("\n[1] Produtos além de investimentos/cartões em cada sessão RF:")
cnt = Counter(tuple(DESVIO(seqs[i])) or ("nenhum",) for i in rf)
for k, v in cnt.most_common():
    print(f"   {' + '.join(k):35s} {v} sessões")
def ordem(s):
    seen = []
    for e in s:
        p = e.split(":")[0]
        if p != "canais" and p not in seen:
            seen.append(p)
    return " → ".join(seen)
print("   ordem dos produtos (exemplos):")
for k, v in Counter(ordem(seqs[i]) for i in rf).most_common(6):
    print(f"     {v}× {k}")

# 2) truncamento
from tokenizers import Tokenizer
tok = Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tok.no_padding(); tok.no_truncation()           # o tokenizer.json vem com padding/truncation ligados
def visiveis_minilm(s, janela=254):           # 256 menos [CLS] e [SEP]
    n, total = 0, 0
    for e in s:
        total += len(tok.encode(e, add_special_tokens=False).ids)
        if total > janela:
            break
        n += 1
    return n
vis = np.array([visiveis_minilm(s) for s in seqs])
toks_gemini = np.array([len(" ".join(s)) / 4 for s in seqs])
print("\n[2] Truncamento")
print(f"   MiniLM: eventos visíveis por sessão — mediana {np.median(vis):.0f} "
      f"(sessões têm mediana de {np.median([len(s) for s in seqs]):.0f} eventos)")
print(f"   MiniLM: sessões truncadas = {np.mean(vis < [len(s) for s in seqs]):.0%} de todas; "
      f"RF truncadas = {np.mean([vis[i] < len(seqs[i]) for i in rf]):.0%}")
RESG_EV, PAG_EV = "investimentos:view_comprovante_resgate", "cartoes:view_comprovante_pagamento"
vr = np.mean([RESG_EV in seqs[i][:vis[i]] for i in rf])
vp = np.mean([PAG_EV in seqs[i][:vis[i]] for i in rf])
print(f"   MiniLM vê o comprovante de resgate em {vr:.0%} e o de pagamento em {vp:.0%} das sessões RF")
print(f"   Gemini (estimativa ~4 car./token, limite 2048): sessões acima do limite = "
      f"{np.mean(toks_gemini > 2048):.0%} de todas; RF = {np.mean(toks_gemini[rf] > 2048):.0%} "
      f"(mediana RF ≈ {np.median(toks_gemini[rf]):.0f} tokens)")

# 3) sensibilidade
def jacc(a, b):
    a, b = set(a), set(b)
    return 1 - len(a & b) / len(a | b) if a | b else 0
pairs = list(combinations(rf, 2))
d_prod = [jacc(prod(seqs[i]), prod(seqs[j])) for i, j in pairs]
d_ev = [jacc(seqs[i], seqs[j]) for i, j in pairs]
d_ini = [jacc(seqs[i][:vis[i]], seqs[j][:vis[j]]) for i, j in pairs]
d_len = [abs(len(seqs[i]) - len(seqs[j])) for i, j in pairs]

spaces = {"TF-IDF": tf["X"]}
for name, path in [("MiniLM", "minilm_embeddings.pkl"), ("Gemini", "gemini_embeddings.pkl")]:
    spaces[name] = np.asarray(pickle.load(open(f"{root}/data/{path}", "rb"))["X"], dtype=np.float32)

first = {i: ordem(seqs[i]).split(" → ")[0] for i in rf}
d_ord = [float(first[i] != first[j]) for i, j in pairs]
sr = lambda a, b: spearmanr(a, b).statistic if np.std(b) > 0 else float("nan")

print("\n[3] Correlação (Spearman) entre distância no espaço e diferenças entre sessões RF")
print(f"   {'espaço':8s} {'produtos':>9s} {'eventos':>9s} {'começo':>9s} {'tamanho':>9s} {'ordem':>9s}   "
      f"percentil médio: mesma ordem × ordem diferente")
for name, X in spaces.items():
    D = pairwise_distances(X)
    iu = np.triu_indices(len(X), 1)
    pct = np.zeros_like(D)
    pct[iu] = rankdata(D[iu]) / len(iu[0])        # percentil da distância entre todos os pares
    pct = pct + pct.T
    d_sp = [pct[i, j] for i, j in pairs]
    r = [sr(d_sp, x) for x in (d_prod, d_ev, d_ini, d_len, d_ord)]
    same = [p for p, do in zip(d_sp, d_ord) if do == 0]
    diff = [p for p, do in zip(d_sp, d_ord) if do > 0]
    print(f"   {name:8s} {r[0]:9.2f} {r[1]:9.2f} {r[2]:9.2f} {r[3]:9.2f} {r[4]:9.2f}   "
          f"{np.mean(same):.0%} ({len(same)} pares) × {np.mean(diff):.0%} ({len(diff)} pares)")
print("   (percentil 10% = entre os 10% de pares mais próximos de todo o conjunto)")
