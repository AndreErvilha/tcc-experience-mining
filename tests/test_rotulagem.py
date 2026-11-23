"""Conferência independente das flags da Tabela 1 revisada, em Python puro, sem pm4py."""
import json
from pathlib import Path

import pandas as pd

from experience_mining.nucleo.rotulo import extract_primary_label
from tests.gerar_referencia import SAIDA as MANIFESTO_PATH, sha256_texto_normalizado

ROOT = Path(__file__).resolve().parent.parent

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
ORDEM = ["001", "002", "003", "004", "005", "012", "006", "007", "008", "009", "010", "011"]


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
            inicio2 = segue(pos, e1[1], G["001"]) if j in ("005", "009") else True
            f[f"is_{suc}"], f[f"is_{ab}"] = int(p1 and p2), int(p1 and inicio2 and not p2)
    return f


def test_conferencia_independente_tabela1(tmp_path):
    ev = pd.read_csv(ROOT / "dataset" / "processed" / "total.csv")
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
        rows.append({
            "cd_sessao": s,
            "primary_label": extract_primary_label(serie),
            "experience": "success" if suc else ("attrition" if ab else "none"),
            **f,
        })

    df = pd.DataFrame(rows)
    out_csv = tmp_path / "conferencia_rotulos.csv"
    df.to_csv(out_csv, index=False)

    manifesto = json.loads(MANIFESTO_PATH.read_text(encoding="utf-8"))
    esperado_hash = manifesto["conferencia_rotulos"]["csv_json_sha256"]["results/dados/conferencia_rotulos.csv"]
    assert sha256_texto_normalizado(out_csv) == esperado_hash

    s = pd.read_csv(ROOT / "data" / "v1.1" / "sessions_represented.csv").set_index("cd_sessao")
    r = df.set_index("cd_sessao").loc[s.index]
    flags = [c for c in r.columns if c.startswith("is_")]
    faltam = [c for c in flags if c not in s.columns]
    assert not faltam, f"Flags ausentes no pipeline: {faltam}"

    assert int((s.primary_label != r.primary_label).sum()) == 0
    assert int((s.experience != r.experience).sum()) == 0
    assert int((s[flags] != r[flags]).any(axis=1).sum()) == 0
