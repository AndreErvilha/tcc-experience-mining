"""Achado 011/012: jornadas avulsas que a Tabela 1 original não previa (v1.1).

Pergunta: as sessões de PIX copia e cola avulso (011, rótulo copia_pix_paga) e de
Open Finance avulso (012, rótulo openfinance_pago) formam grupo próprio sem
supervisão? E uma IA, sem ver o gabarito, dá a esses grupos um nome coerente?

Também mede a sensibilidade: ARI e V-Measure com e sem as sessões 011/012
(para mostrar que incluí-las no gabarito não infla as métricas).

Entradas:
- analises/v1.1/saidas/sessoes_por_modelo.csv (montar_tabela_sessoes.py)
- .ia-rotulos/v1.1/<modelo>.json (rotulagem cega por IA; pasta fora do git)
Saída: analises/v1.1/saidas/achado_011_012.csv e sensibilidade_011_012.csv
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import adjusted_rand_score, v_measure_score

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
MODELOS = ["kmeans_tf_idf", "kmeans_minilm", "kmeans_gemini",
           "hdbscan_tf_idf", "hdbscan_minilm", "hdbscan_gemini"]
JORNADAS = {"011": "copia_pix_paga", "012": "openfinance_pago",
            "010": "fatura_paga_pix_copia_cola", "005": "openfinance_fatura_paga"}


def main():
    s = pd.read_csv(AQUI / "saidas" / "sessoes_por_modelo.csv", sep=";")
    ia = {}
    for m in MODELOS:
        f = RAIZ / ".ia-rotulos" / "v1.1" / f"{m}.json"
        ia[m] = {g["grupo"]: g for g in json.loads(f.read_text(encoding="utf-8"))["grupos"]} if f.exists() else {}

    linhas = []
    for jid, rot in JORNADAS.items():
        alvo = s[s.primary_label == rot]
        for m in MODELOS:
            vc = alvo[m].value_counts()
            g, n = vc.index[0], int(vc.iloc[0])
            tam = int((s[m] == g).sum())
            linhas.append({
                "jornada": jid, "rotulo": rot, "modelo": m, "sessoes_jornada": len(alvo),
                "grupo_principal": g, "sessoes_no_grupo": n, "tamanho_grupo": tam,
                "pct_da_jornada": round(n / len(alvo), 3), "pct_do_grupo": round(n / tam, 3),
                "nome_ia_cega": ia[m].get(g, {}).get("rotulo", ""),
            })
    out = pd.DataFrame(linhas)
    out.to_csv(AQUI / "saidas" / "achado_011_012.csv", sep=";", index=False)
    print(out.to_string(index=False))

    sens = []
    for m in MODELOS:
        d = s[s[m] != "ruido"]
        e = d[~d.primary_label.isin([JORNADAS["011"], JORNADAS["012"]])]
        sens.append({
            "modelo": m, "n_com": len(d), "n_sem": len(e),
            "ari_com": round(adjusted_rand_score(d.primary_label, d[m]), 4),
            "ari_sem_011_012": round(adjusted_rand_score(e.primary_label, e[m]), 4),
            "vm_com": round(v_measure_score(d.primary_label, d[m]), 4),
            "vm_sem_011_012": round(v_measure_score(e.primary_label, e[m]), 4),
        })
    sens = pd.DataFrame(sens)
    sens.to_csv(AQUI / "saidas" / "sensibilidade_011_012.csv", sep=";", index=False)
    print(sens.to_string(index=False))


if __name__ == "__main__":
    main()
