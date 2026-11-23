"""Conferência independente das flags da Tabela 1 revisada, em Python puro, sem pm4py.

O rótulo oficial vem do dataset/gerar_massas.py (pm4py-cli) -> dataset/processed/total-rotulado-v1.1.csv
-> src/events2vector.py -> data/sessions_represented.csv. Este script refaz as regras direto nos eventos
e confere flags, rótulo principal e experience com o data/sessions_represented.csv.

Regras (mesma semântica do pm4py-cli: cada par "A e, depois, B" vale por si):
  - simples: sucesso = (gatilho, conversão); abandono = gatilho e não (gatilho, conversão);
  - composta: sucesso = (etapa 1) e (etapa 2); abandono = (etapa 1), início da etapa 2 e não (etapa 2).
Saída: results/dados/conferencia_rotulos.csv
Uso (da raiz do repositório): python analises/conferir_rotulo_tabela1.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from events2vector import extract_primary_label  # noqa: E402

G = {"001": "cartoes:fatura:fatura_atual.click_button(opcoes_pagamento)",
     "002": "cartoes:pagamento:opcoes_pagamento.click_button(copiar_pix)",
     "003": "cartoes:fatura:fatura_atual.click_button(simular_parcelamento)",
     "004": "cartoes:historico:historico_faturas.click_button(mes_anterior)",
     "012": "open_finance:home:home_open_finance.click_button(trazer_dinheiro)",
     "006": "pix:transferencia:home_pix.click_button(transferir_chave)",
     "007": "investimentos:renda_fixa:home_investimentos.click_button(renda_fixa)",
     "008": "investimentos:meus_investimentos:carteira_renda_fixa.click_button(detalhar_cdb)",
     "011": "pix:home:home_pix.click_button(pix_copia_e_cola)"}
C = {"001": "cartoes:pagamento:comprovante_pagamento.page_view()",
     "002": "cartoes:pagamento:sucesso_copia.page_view()",
     "003": "cartoes:parcelamento:comprovante_parcelamento.page_view()",
     "004": "cartoes:historico:visualizacao_pdf.page_view()",
     "012": "open_finance:trazer_dinheiro:comprovante_pix_open_finance.page_view()",
     "006": "pix:transferencia:comprovante_pix.page_view()",
     "007": "investimentos:renda_fixa:comprovante_investimento.page_view()",
     "008": "investimentos:resgate:comprovante_resgate.page_view()",
     "011": "pix:pix_copia_e_cola:comprovante_pix.page_view()"}
SIMPLES = {"001": ("fatura_paga_saldo", "iniciou_pagar_saldo_sem_converter"),
           "002": ("pix_copiado", "iniciou_copiar_pix_sem_converter"),
           "003": ("fatura_parcelada", "iniciou_parcelamento_sem_converter"),
           "004": ("historico_visualizado", "iniciou_historico_sem_converter"),
           "012": ("openfinance_pago", "iniciou_openfinance_sem_converter"),
           "006": ("pix_transferido", "iniciou_pix_sem_converter"),
           "007": ("cdb_aplicado", "iniciou_cdb_sem_converter"),
           "008": ("cdb_resgatado", "iniciou_resgate_sem_converter"),
           "011": ("copia_pix_paga", "iniciou_copia_pix_sem_converter")}
COMPOSTAS = {"005": ("openfinance_fatura_paga", "iniciou_openfinance_fatura_sem_converter",
                     (G["012"], C["012"]), (C["012"], C["001"])),
             "009": ("resgate_pago_fatura", "iniciou_resgate_pagar_fatura_sem_converter",
                     (G["008"], C["008"]), (C["008"], C["001"])),
             "010": ("fatura_paga_pix_copia_cola", "iniciou_fatura_pix_copia_cola_sem_converter",
                     (G["001"], G["011"]), (G["011"], C["011"]))}
ORDEM = ["001", "002", "003", "004", "005", "012", "006", "007", "008", "009", "010", "011"]  # ordem das colunas


def segue(pos, a, b):
    """Existe A e, depois, B."""
    return bool(pos.get(a)) and bool(pos.get(b)) and min(pos[a]) < max(pos[b])


def flags_da_sessao(tokens):
    pos = {}
    for i, e in enumerate(tokens):
        pos.setdefault(e, []).append(i)
    f = {}
    for j in ORDEM:
        if j in SIMPLES:
            suc, ab = SIMPLES[j]
            ok = segue(pos, G[j], C[j])
            f[f"is_{suc}"], f[f"is_{ab}"] = int(ok), int(bool(pos.get(G[j])) and not ok)
        else:
            suc, ab, e1, e2 = COMPOSTAS[j]
            p1, p2 = segue(pos, *e1), segue(pos, *e2)
            # abandono: etapa 1, início da etapa 2 (005 e 009: opções de pagamento depois) e sem a etapa 2
            inicio2 = segue(pos, e1[1], G["001"]) if j in ("005", "009") else True
            f[f"is_{suc}"], f[f"is_{ab}"] = int(p1 and p2), int(p1 and inicio2 and not p2)
    return f


def main():
    ev = pd.read_csv("dataset/processed/total.csv")
    ev["ts"] = pd.to_datetime(ev["timestamp"])
    ev = ev.sort_values(["cd_sessao", "ts"], kind="stable")
    btn = ev.button_text.fillna("").astype(str)
    ev["tok"] = ev.cd_produto + ":" + ev.cd_sub_produto + ":" + ev.page_name + "." + ev.cd_evento_tipo + "(" + btn + ")"
    rows = []
    for s, g in ev.groupby("cd_sessao", sort=False):
        f = flags_da_sessao(g.tok.tolist())
        serie = pd.Series(f)
        suc = any(v for k, v in f.items() if "sem_converter" not in k)
        ab = any(v for k, v in f.items() if "sem_converter" in k)
        rows.append({"cd_sessao": s, "primary_label": extract_primary_label(serie),
                     "experience": "success" if suc else ("attrition" if ab else "none"), **f})
    df = pd.DataFrame(rows)
    os.makedirs("results/dados", exist_ok=True)
    df.to_csv("results/dados/conferencia_rotulos.csv", index=False)
    s = pd.read_csv("data/sessions_represented.csv").set_index("cd_sessao")
    r = df.set_index("cd_sessao").loc[s.index]
    flags = [c for c in r.columns if c.startswith("is_")]
    faltam = [c for c in flags if c not in s.columns]
    if faltam:
        raise SystemExit(f"[!] Flags ausentes no pipeline: {faltam}")
    dif = {"primary_label": int((s.primary_label != r.primary_label).sum()),
           "experience": int((s.experience != r.experience).sum()),
           "flags": int((s[flags] != r[flags]).any(axis=1).sum())}
    print("Sessões diferentes do data/sessions_represented.csv:", dif)
    if any(dif.values()):
        raise SystemExit("[!] O rótulo do pipeline não bate com a regra da Tabela 1 revisada.")


if __name__ == "__main__":
    main()
