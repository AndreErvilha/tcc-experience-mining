"""Testes do núcleo de jornadas, rotulagem e métricas."""
import json
from pathlib import Path

import pandas as pd

from experience_mining.io import carregar_agrupamento_pkl, ler_json, salvar_csv
from experience_mining.nucleo.jornadas import JORNADAS, MODELOS
from experience_mining.nucleo.metricas import (
    avaliar_agrupamento,
    calcular_cobertura_jornadas,
    calcular_contagem_jornadas,
)
from experience_mining.nucleo.rotulo import extract_primary_label
from tests.gerar_referencia import SAIDA as MANIFESTO_PATH, sha256_texto_normalizado

ROOT = Path(__file__).resolve().parent.parent


def _serie_flags(**ativas) -> pd.Series:
    cols = {}
    for j in JORNADAS:
        cols[f"is_{j.flag_sucesso}"] = 0
        cols[f"is_{j.flag_abandono}"] = 0
    for k, v in ativas.items():
        cols[k] = v
    return pd.Series(cols)


def test_extract_primary_label_regras():
    assert extract_primary_label(_serie_flags(is_fatura_paga_saldo=1)) == "fatura_paga_saldo"

    # Composta 009 absorve 008 e 001
    comp009 = _serie_flags(is_resgate_pago_fatura=1, is_cdb_resgatado=1, is_fatura_paga_saldo=1)
    assert extract_primary_label(comp009) == "resgate_pago_fatura"

    # Composta 005 absorve 012 e 001
    comp005 = _serie_flags(is_openfinance_fatura_paga=1, is_openfinance_pago=1, is_fatura_paga_saldo=1)
    assert extract_primary_label(comp005) == "openfinance_fatura_paga"

    # Composta 010 absorve 011
    comp010 = _serie_flags(is_fatura_paga_pix_copia_cola=1, is_copia_pix_paga=1)
    assert extract_primary_label(comp010) == "fatura_paga_pix_copia_cola"

    # Abandono composto absorve abandonos simples
    ab005 = _serie_flags(
        is_iniciou_openfinance_fatura_sem_converter=1,
        is_iniciou_openfinance_sem_converter=1,
        is_iniciou_pagar_saldo_sem_converter=1,
    )
    assert extract_primary_label(ab005) == "iniciou_openfinance_fatura_sem_converter"

    # Duas simples sem composta que as cubra
    dupla = _serie_flags(is_fatura_paga_saldo=1, is_cdb_resgatado=1)
    assert extract_primary_label(dupla) == "fatura_paga_saldo+cdb_resgatado"

    # Sem sucesso -> abandono ou neutro
    assert extract_primary_label(_serie_flags(is_iniciou_pix_sem_converter=1)) == "iniciou_pix_sem_converter"
    assert extract_primary_label(_serie_flags()) == "neutro"


def test_metricas_nucleo_batem_com_referencia(tmp_path):
    manifesto = ler_json(MANIFESTO_PATH)["pipeline"]["csv_json_sha256"]
    rot = pd.read_csv(ROOT / "data" / "v1.1" / "sessions_represented.csv").set_index("cd_sessao")
    clusters_por_modelo = {}
    for m in MODELOS:
        cl = carregar_agrupamento_pkl(ROOT / "clusters" / "v1.0" / f"{m}.pkl")
        clusters_por_modelo[m] = cl
        ids = list(cl["session_ids"])
        cl_eval = dict(cl)
        cl_eval["primary_labels"] = rot.loc[ids, "primary_label"].tolist()
        cl_eval["experiences"] = rot.loc[ids, "experience"].tolist()

        obtido = avaliar_agrupamento(cl_eval)
        esperado = ler_json(ROOT / "results" / "v1.1" / "metrics" / f"metrics_{m}.json")
        assert json.loads(json.dumps(obtido)) == esperado, f"Divergencia em {m}"

    sessions_df = pd.read_csv(ROOT / "data" / "v1.1" / "sessions_represented.csv")
    contagem_csv = salvar_csv(calcular_contagem_jornadas(sessions_df), tmp_path / "jornadas_contagem.csv", ptbr=True)
    assert sha256_texto_normalizado(contagem_csv) == manifesto["results/dados/jornadas_contagem.csv"]

    cobertura_csv = salvar_csv(
        calcular_cobertura_jornadas(sessions_df, clusters_por_modelo),
        tmp_path / "cobertura_por_jornada.csv",
        ptbr=True,
    )
    assert sha256_texto_normalizado(cobertura_csv) == manifesto["results/metrics/cobertura_por_jornada.csv"]
