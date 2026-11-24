"""Monta uma tabela única sessão × grupo de cada modelo (v1.1).

Lê results/v1.1/jornadas_por_grupo/<modelo>/grupos/<grupo>/sessoes.csv
e grava analises/v1.1/saidas/sessoes_por_modelo.csv.
Base para as análises de apoio às hipóteses (não altera nenhum resultado).
"""
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
BASE = RAIZ / "results" / "v1.1" / "jornadas_por_grupo"
SAIDA = Path(__file__).resolve().parent / "saidas"
MODELOS = ["kmeans_tf_idf", "kmeans_minilm", "kmeans_gemini",
           "hdbscan_tf_idf", "hdbscan_minilm", "hdbscan_gemini"]

def main():
    base = None
    for m in MODELOS:
        partes = []
        for f in sorted((BASE / m / "grupos").glob("*/sessoes.csv")):
            d = pd.read_csv(f, sep=";", encoding="utf-8-sig")
            d[m] = f.parent.name
            partes.append(d)
        d = pd.concat(partes, ignore_index=True)
        cols = ["cd_sessao", "primary_label", "experience", "num_events", "text_representation"]
        if base is None:
            base = d[cols + [m]]
        else:
            base = base.merge(d[["cd_sessao", m]], on="cd_sessao", how="outer")
    SAIDA.mkdir(exist_ok=True)
    base.to_csv(SAIDA / "sessoes_por_modelo.csv", index=False, sep=";")
    print(base.shape, base[MODELOS].nunique().to_dict())

if __name__ == "__main__":
    main()
