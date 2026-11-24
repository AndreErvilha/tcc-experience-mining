"""Atrito invisível ao funil: mesma jornada, mesma conversão, esforço diferente (v1.1).

Procura, em cada modelo, pares de grupos dominados pela MESMA jornada (>= 85%),
ambos com sucesso alto (>= 85%), mas com esforço (mediana de eventos) muito
diferente. Para o grupo mais longo, mostra o evento que o SHAP aponta como
marcador (maior SHAP médio no grupo), com % dentro e fora do grupo.

Um funil diria "converteu" para os dois; o agrupamento mostra que um deles
converte com várias vezes mais esforço, e o SHAP diz onde.

Entradas: results/v1.1/jornadas_por_grupo/resumo_grupos.csv e shap_pesos_todos_modelos.csv
Saída: analises/v1.1/saidas/atrito_invisivel.csv
"""
from pathlib import Path
import pandas as pd

AQUI = Path(__file__).resolve().parent
BASE = AQUI.parents[1] / "results" / "v1.1" / "jornadas_por_grupo"
MIN_DOM, MIN_SUC, MIN_RAZAO = 0.85, 0.85, 2.0


def main():
    r = pd.read_csv(BASE / "resumo_grupos.csv", sep=";", encoding="utf-8-sig")
    w = pd.read_csv(BASE / "shap_pesos_todos_modelos.csv", sep=";", encoding="utf-8-sig")
    r = r[(r.grupo.astype(str) != "-1") & (r.pct_jornada_dominante >= MIN_DOM) & (r.pct_sucesso >= MIN_SUC)]
    linhas = []
    for (m, j), d in r.groupby(["modelo", "jornada_dominante"]):
        if len(d) < 2:
            continue
        curto = d.loc[d.mediana_eventos.idxmin()]
        longo = d.loc[d.mediana_eventos.idxmax()]
        razao = longo.mediana_eventos / curto.mediana_eventos
        if razao < MIN_RAZAO:
            continue
        sw = w[(w.modelo == m) & (w.grupo.astype(str) == str(longo.grupo))]
        top = sw.sort_values("shap_medio_no_grupo", ascending=False).iloc[0] if len(sw) else None
        linhas.append({
            "modelo": m, "jornada": j,
            "grupo_curto": curto.grupo, "sessoes_curto": curto.sessoes, "mediana_eventos_curto": curto.mediana_eventos,
            "sucesso_curto": curto.pct_sucesso,
            "grupo_longo": longo.grupo, "sessoes_longo": longo.sessoes, "mediana_eventos_longo": longo.mediana_eventos,
            "sucesso_longo": longo.pct_sucesso, "razao_esforco": round(razao, 1),
            "marcador_shap_longo": None if top is None else top.evento,
            "pct_no_grupo": None if top is None else top.pct_sessoes_do_grupo_com_evento,
            "pct_fora": None if top is None else top.pct_sessoes_fora_com_evento,
        })
    out = pd.DataFrame(linhas).sort_values("razao_esforco", ascending=False)
    (AQUI / "saidas").mkdir(exist_ok=True)
    out.to_csv(AQUI / "saidas" / "atrito_invisivel.csv", sep=";", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
