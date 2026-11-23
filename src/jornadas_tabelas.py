"""Tabelas por jornada (Tabela 1 revisada), a partir do rótulo do pipeline.

  results/dados/jornadas_contagem.csv      por jornada: sessões do cenário de origem, quantas concluíram,
                                           quantas não concluíram (abandono) e quantas têm a jornada como rótulo principal
  results/metrics/cobertura_por_jornada.csv  por jornada e modelo: % das sessões da jornada em grupos com
                                           pelo menos 80% dela (aceita subdividir em vários grupos puros)

Entradas: data/sessions_represented.csv (flags is_* e rótulo) e clusters/<modelo>.pkl.
Uso: python src/jornadas_tabelas.py
"""
import os
import pickle

import numpy as np
import pandas as pd

# id, nome, flag de sucesso, flag de abandono, cenário de origem (modelo PNML)
JORNADAS = [
    ("001", "Pagar fatura com saldo", "fatura_paga_saldo", "iniciou_pagar_saldo_sem_converter", "001"),
    ("002", "Copiar o PIX da fatura", "pix_copiado", "iniciou_copiar_pix_sem_converter", "002"),
    ("003", "Parcelar a fatura", "fatura_parcelada", "iniciou_parcelamento_sem_converter", "003"),
    ("004", "Histórico de faturas em PDF", "historico_visualizado", "iniciou_historico_sem_converter", "004"),
    ("005", "Trazer dinheiro (Open Finance) e pagar a fatura", "openfinance_fatura_paga", "iniciou_openfinance_fatura_sem_converter", "005"),
    ("006", "PIX por chave", "pix_transferido", "iniciou_pix_sem_converter", "006"),
    ("007", "Aplicar em CDB", "cdb_aplicado", "iniciou_cdb_sem_converter", "007"),
    ("008", "Resgatar CDB", "cdb_resgatado", "iniciou_resgate_sem_converter", "008"),
    ("009", "Resgatar CDB e pagar a fatura", "resgate_pago_fatura", "iniciou_resgate_pagar_fatura_sem_converter", "009"),
    ("010", "Ver a fatura e pagar pelo PIX copia e cola", "fatura_paga_pix_copia_cola", "iniciou_fatura_pix_copia_cola_sem_converter", "010"),
    ("011", "PIX copia e cola avulso", "copia_pix_paga", "iniciou_copia_pix_sem_converter", "010"),
    ("012", "Trazer dinheiro (Open Finance) avulso", "openfinance_pago", "iniciou_openfinance_sem_converter", "005"),
]
# Até a v1.0 (dataset/gerar_massas.py entregue) eram 10 jornadas; duas flags mudaram de jornada na v1.1:
# openfinance_pago era a 005 (hoje 012) e copia_pix_paga era a 010 (hoje 011).
# flags da v1.0 (sucesso, abandono) por id: 10 jornadas, só o evento de conversão
IDS_V10 = {f: j for j, par in {
    "001": ("fatura_paga_saldo", "iniciou_pagar_saldo_sem_converter"),
    "002": ("pix_copiado", "iniciou_copiar_pix_sem_converter"),
    "003": ("fatura_parcelada", "iniciou_parcelamento_sem_converter"),
    "004": ("historico_visualizado", "iniciou_historico_sem_converter"),
    "005": ("openfinance_pago", "iniciou_openfinance_sem_converter"),
    "006": ("pix_transferido", "iniciou_pix_sem_converter"),
    "007": ("cdb_aplicado", "iniciou_cdb_sem_converter"),
    "008": ("cdb_resgatado", "iniciou_resgate_sem_converter"),
    "009": ("resgate_pago_fatura", "iniciou_resgate_pagar_fatura_sem_converter"),
    "010": ("copia_pix_paga", "iniciou_copia_pix_sem_converter")}.items() for f in par}
IDS_V11 = {f: j for j, _, suc, ab, _ in JORNADAS for f in (suc, ab)}


def tipo_rotulo(rotulo):
    """0 = sucesso, 1 = atrito (abandono), 2 = neutro."""
    if rotulo == "neutro":
        return 2
    return 1 if "sem_converter" in rotulo else 0


def codigo(rotulo, ids=IDS_V11):
    """'fatura_paga_saldo+openfinance_pago' -> '001+012 sucesso'; 'iniciou_pagar_saldo_sem_converter' -> '001 atrito'."""
    if rotulo == "neutro":
        return "neutro"
    return "+".join(ids.get(p, p) for p in rotulo.split("+")) + (" atrito" if tipo_rotulo(rotulo) == 1 else " sucesso")


def colunas_completas(presentes, versao="v1.1"):
    """Todas as jornadas da versão (sucesso ao lado do atrito, mesmo sem sessões), os rótulos com '+' de cada
    jornada logo depois dela e 'neutro' por último."""
    ids = IDS_V11 if versao == "v1.1" else IDS_V10
    base = list(ids)
    extras = [r for r in presentes if r not in base and r != "neutro"]
    chave = lambda r: (codigo(r, ids).split()[0], tipo_rotulo(r))
    return sorted(base + extras, key=chave) + ["neutro"]


MODELOS = ["kmeans_tf_idf", "kmeans_minilm", "kmeans_gemini", "hdbscan_tf_idf", "hdbscan_minilm", "hdbscan_gemini"]
CSV = dict(sep=";", index=False, encoding="utf-8-sig")


def main():
    s = pd.read_csv("data/sessions_represented.csv")
    cen = ((s.cd_sessao.str[5:].astype(int) - 1) // 100 + 1).map(lambda c: f"{c:03d}")  # 100 ids por cenário no gerador
    linhas = [{"jornada": j, "nome": n, "cenario_origem": c, "sessoes_do_cenario": int((cen == c).sum()),
               "concluiram": int(s[f"is_{suc}"].sum()), "nao_concluiram": int(s[f"is_{ab}"].sum()),
               "rotulo_principal": int((s.primary_label == suc).sum())} for j, n, suc, ab, c in JORNADAS]
    os.makedirs("results/dados", exist_ok=True)
    pd.DataFrame(linhas).to_csv("results/dados/jornadas_contagem.csv", **CSV)

    y = s.set_index("cd_sessao").primary_label
    cats = [c for c, n in y.value_counts().items() if n >= 10]
    cob = []
    for c in cats:
        r = {"rotulo": c, "sessoes": int((y == c).sum())}
        for m in MODELOS:
            cl = pickle.load(open(f"clusters/{m}.pkl", "rb"))
            lab = np.asarray(cl["cluster_labels"])
            inj = (y.loc[list(cl["session_ids"])] == c).values
            puros = [k for k in set(lab) - {-1} if np.mean(inj[lab == k]) >= 0.8]
            r[m] = round(float(np.sum(inj & np.isin(lab, puros)) / inj.sum() * 100), 1)
        cob.append(r)
    cob = pd.DataFrame(cob)
    total = {"rotulo": "jornadas_com_cobertura_>=80%", "sessoes": len(cats), **{m: int((cob[m] >= 80).sum()) for m in MODELOS}}
    cob = pd.concat([cob, pd.DataFrame([total])])
    os.makedirs("results/metrics", exist_ok=True)
    cob.to_csv("results/metrics/cobertura_por_jornada.csv", **CSV)
    print(pd.DataFrame(linhas).to_string(index=False))
    print(cob.to_string(index=False))


if __name__ == "__main__":
    main()
