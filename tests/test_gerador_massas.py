"""Testes do gerador e rotulador guiados por configuracao JSON (gerador-de-massas)."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import dataset.gerar_massas as gerar_massas
from dataset.gerar_massas import gerar_eventos, resolver_parametros_cenarios, rotular_massa
from experience_mining.configuracao import (
    ConfigEventos,
    ConfigRotulagem,
    ErroConfiguracao,
    carregar_config_json,
)
from experience_mining.io import validar_contrato_eventos_csv, validar_contrato_rotulo_csv
from experience_mining.orquestracao import _preparar_sessoes
from tests.test_rotulagem import flags_da_sessao

ROOT = Path(__file__).resolve().parent.parent


def _ler_v2_cfg() -> dict:
    return json.loads((ROOT / "configs" / "v2.0.json").read_text(encoding="utf-8"))


def test_sobrescrita_por_cenario_e_relatorio_min_events(tmp_path):
    raw = _ler_v2_cfg()
    selecionados = [copy.deepcopy(c) for c in raw["events"]["events_params"] if c["id"] in ("001", "005", "010")]
    for c in selecionados:
        if c["id"] == "010":
            c["num_traces"] = 80
    raw["events"]["events_params"] = selecionados
    raw["events"]["output_events_csv_path"] = str(tmp_path / "total.csv")
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")

    cfg = carregar_config_json(cfg_path, root=ROOT, validar_contratos=True)
    resolvidos = resolver_parametros_cenarios(cfg.eventos)
    por_id = {c["id"]: c for c in resolvidos}
    assert por_id["001"]["num_traces"] == 100
    assert por_id["005"]["num_traces"] == 100
    assert por_id["010"]["num_traces"] == 80

    rel = gerar_eventos(cfg.eventos, root=ROOT, work_dir=tmp_path / "_work")
    assert rel["relatorio_csv_path"].exists()
    df_rel = pd.read_csv(rel["relatorio_csv_path"], dtype={"cenario_id": str}).set_index("cenario_id")

    assert int(df_rel.loc["001", "sessoes_simuladas"]) == 100
    assert int(df_rel.loc["005", "sessoes_simuladas"]) == 100
    assert int(df_rel.loc["010", "sessoes_simuladas"]) == 80
    for cid in ("001", "005", "010"):
        assert int(df_rel.loc[cid, "removidas_min_events"]) == (
            int(df_rel.loc[cid, "sessoes_simuladas"]) - int(df_rel.loc[cid, "sessoes_finais"])
        )

    df_total = pd.read_csv(rel["output_events_csv_path"])
    assert int(df_rel["sessoes_finais"].sum()) == df_total["cd_sessao"].nunique()

    # Sessoes do cenario 005 (SESS_101..SESS_200) respeitam min_events=15
    ids_005 = {f"SESS_{i:03d}" for i in range(101, 201)}
    sub_005 = df_total[df_total["cd_sessao"].isin(ids_005)]
    if len(sub_005) > 0:
        assert int(sub_005.groupby("cd_sessao").size().min()) >= 15


def test_semente_ausente_para_antes_do_primeiro_playout(tmp_path):
    raw = _ler_v2_cfg()
    raw["events"]["events_params"] = [
        copy.deepcopy(raw["events"]["events_params"][0]),
        {"id": "002", "seed_csv_path": "dataset/raw/inexistente/nao_existe.csv", "min_events": 7},
    ]
    cfg_path = tmp_path / "cfg_err.json"
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ErroConfiguracao, match="002"):
        carregar_config_json(cfg_path, root=ROOT, validar_contratos=True)

    workdir = tmp_path / "_work"
    ev_cfg = ConfigEventos(
        events_csv_path=tmp_path / "total.csv",
        output_events_csv_path=tmp_path / "total.csv",
        rotular_massa=False,
        massa_rotulada_csv_path=tmp_path / "total-rotulado.csv",
        sessions_csv_path=tmp_path / "sessions.csv",
        modo="gerar",
        defaults={"num_traces": 5, "seed": 42},
        events_params=raw["events"]["events_params"],
    )
    with pytest.raises(ErroConfiguracao, match="002"):
        gerar_eventos(ev_cfg, root=ROOT, work_dir=workdir)
    assert not workdir.exists()


def test_rotulagem_total_golden_master_zero_diferencas_e_absorbs(tmp_path):
    cfg = carregar_config_json(ROOT / "configs" / "v2.0.json", root=ROOT, validar_contratos=False)
    out_rot = rotular_massa(
        cfg.labeling,
        ROOT / "dataset" / "processed" / "total.csv",
        root=ROOT,
        output_csv_override=tmp_path / "total-rotulado.csv",
    )
    df_ger = pd.read_csv(out_rot).set_index("cd_sessao")
    assert len(df_ger) == 516

    # Golden master contra dataset/processed/total-rotulado-v1.1.csv (24 flags is_*)
    df_v11 = pd.read_csv(ROOT / "dataset" / "processed" / "total-rotulado-v1.1.csv")
    is_cols = [c for c in df_ger.columns if c.startswith("is_")]
    assert len(is_cols) == 24
    v11_sess = df_v11.groupby("cd_sessao")[is_cols].max().loc[df_ger.index]
    assert int((df_ger[is_cols] != v11_sess[is_cols]).any(axis=1).sum()) == 0

    # Comparacao com conferencia independente (tests/test_rotulagem.py) e data/v1.1/sessions_represented.csv
    df_ev = pd.read_csv(ROOT / "dataset" / "processed" / "total.csv")
    df_ev["ts"] = pd.to_datetime(df_ev["timestamp"])
    df_ev = df_ev.sort_values(["cd_sessao", "ts"], kind="stable")
    btn = df_ev["button_text"].fillna("").astype(str)
    df_ev["tok"] = (
        df_ev["cd_produto"]
        + ":"
        + df_ev["cd_sub_produto"]
        + ":"
        + df_ev["page_name"]
        + "."
        + df_ev["cd_evento_tipo"]
        + "("
        + btn
        + ")"
    )
    ind_rows = []
    for sess, grp in df_ev.groupby("cd_sessao", sort=False):
        f = flags_da_sessao(grp["tok"].tolist())
        ind_rows.append({"cd_sessao": sess, **f})
    df_ind = pd.DataFrame(ind_rows).set_index("cd_sessao").loc[df_ger.index]
    assert int((df_ger[is_cols] != df_ind[is_cols]).any(axis=1).sum()) == 0

    df_sess_v11 = pd.read_csv(ROOT / "data" / "v1.1" / "sessions_represented.csv").set_index("cd_sessao").loc[df_ger.index]
    assert int((df_ger["primary_label"] != df_sess_v11["primary_label"]).sum()) == 0
    assert int((df_ger["experience"] != df_sess_v11["experience"]).sum()) == 0
    assert int((df_ger[is_cols] != df_sess_v11[is_cols]).any(axis=1).sum()) == 0

    # Regra sintetica com absorbs_label_cols (sucesso composto e abandono composto)
    ev_sint = tmp_path / "ev_sint.csv"
    pd.DataFrame(
        [
            {"cd_sessao": "S1", "cd_produto": "p", "cd_sub_produto": "s", "page_name": "a", "cd_evento_tipo": "page_view", "button_text": "", "timestamp": "2025-01-01 00:00:01"},
            {"cd_sessao": "S1", "cd_produto": "p", "cd_sub_produto": "s", "page_name": "b", "cd_evento_tipo": "page_view", "button_text": "", "timestamp": "2025-01-01 00:00:02"},
            {"cd_sessao": "S2", "cd_produto": "p", "cd_sub_produto": "s", "page_name": "x", "cd_evento_tipo": "page_view", "button_text": "", "timestamp": "2025-01-01 00:00:01"},
            {"cd_sessao": "S2", "cd_produto": "p", "cd_sub_produto": "s", "page_name": "y", "cd_evento_tipo": "page_view", "button_text": "", "timestamp": "2025-01-01 00:00:02"},
        ]
    ).to_csv(ev_sint, index=False)
    lbl_custom = ConfigRotulagem(
        modo="gerar",
        output_labeling_csv_path=tmp_path / "rot_sint.csv",
        labeling_params=[
            {
                "label_col": "is_comp_custom",
                "pattern": [["[p:s:a].page_view()", "[p:s:b].page_view()"]],
                "absorbs_label_cols": ["is_simples_custom"],
            },
            {"label_col": "is_simples_custom", "contain": ["[p:s:a].page_view()"]},
            {
                "label_col": "is_ab_comp_sem_converter",
                "pattern": [["[p:s:x].page_view()", "[p:s:y].page_view()"]],
                "absorbs_label_cols": ["is_ab_simp_sem_converter"],
            },
            {"label_col": "is_ab_simp_sem_converter", "contain": ["[p:s:x].page_view()"]},
        ],
    )
    out_sint = pd.read_csv(rotular_massa(lbl_custom, ev_sint, root=ROOT)).set_index("cd_sessao")
    assert out_sint.loc["S1", "primary_label"] == "comp_custom"
    assert out_sint.loc["S1", "experience"] == "success"
    assert out_sint.loc["S2", "primary_label"] == "ab_comp_sem_converter"
    assert out_sint.loc["S2", "experience"] == "attrition"


def test_sem_cenarios_config_nem_flags_cli_e_v2_completo(tmp_path):
    assert not hasattr(gerar_massas, "CENARIOS_CONFIG")
    assert "argparse" not in (ROOT / "dataset" / "gerar_massas.py").read_text(encoding="utf-8")

    res_cli = subprocess.run(
        [sys.executable, str(ROOT / "dataset" / "gerar_massas.py"), "--skip_playout"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert res_cli.returncode != 0

    raw = _ler_v2_cfg()
    raw["output_dir_path"] = str(tmp_path / "results")
    raw["events"]["output_events_csv_path"] = str(tmp_path / "dataset" / "total.csv")
    raw["labeling"]["output_labeling_csv_path"] = str(tmp_path / "dataset" / "total-rotulado.csv")
    cfg_path = tmp_path / "v2.json"
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")

    cfg = carregar_config_json(cfg_path, root=ROOT, validar_contratos=True)
    cfg.eventos.sessions_csv_path = tmp_path / "data" / "sessions_represented.csv"

    rel = gerar_eventos(cfg.eventos, root=ROOT, work_dir=tmp_path / "_work")
    assert len(rel["cenarios"]) == 10
    event_ids = validar_contrato_eventos_csv(cfg.eventos.events_csv_path)

    out_lbl = rotular_massa(cfg.labeling, cfg.eventos.events_csv_path, root=ROOT)
    df_lbl = validar_contrato_rotulo_csv(out_lbl, event_session_ids=event_ids).set_index("cd_sessao")

    _preparar_sessoes(cfg)
    assert cfg.eventos.sessions_csv_path.exists()
    df_sess = pd.read_csv(cfg.eventos.sessions_csv_path).set_index("cd_sessao")
    assert len(df_sess) == len(df_lbl)

    # Conferencia independente na massa nova v2 gerada
    df_ev = pd.read_csv(cfg.eventos.events_csv_path)
    df_ev["ts"] = pd.to_datetime(df_ev["timestamp"])
    df_ev = df_ev.sort_values(["cd_sessao", "ts"], kind="stable")
    btn = df_ev["button_text"].fillna("").astype(str)
    df_ev["tok"] = (
        df_ev["cd_produto"]
        + ":"
        + df_ev["cd_sub_produto"]
        + ":"
        + df_ev["page_name"]
        + "."
        + df_ev["cd_evento_tipo"]
        + "("
        + btn
        + ")"
    )
    is_cols = [c for c in df_lbl.columns if c.startswith("is_")]
    ind_rows = [{"cd_sessao": s, **flags_da_sessao(g["tok"].tolist())} for s, g in df_ev.groupby("cd_sessao", sort=False)]
    df_ind = pd.DataFrame(ind_rows).set_index("cd_sessao").loc[df_lbl.index]
    assert int((df_lbl[is_cols] != df_ind[is_cols]).any(axis=1).sum()) == 0
    assert int((df_sess["primary_label"] != df_lbl["primary_label"]).sum()) == 0
    assert int((df_sess["experience"] != df_lbl["experience"]).sum()) == 0


def test_determinismo_playout_v2_e_pm4py_cli(tmp_path):
    raw = _ler_v2_cfg()
    selecionados = [copy.deepcopy(c) for c in raw["events"]["events_params"] if c["id"] in ("001", "005")]

    saidas = []
    for run_id in ("r1", "r2"):
        rdir = tmp_path / run_id
        ev_cfg = ConfigEventos(
            events_csv_path=rdir / "total.csv",
            output_events_csv_path=rdir / "total.csv",
            rotular_massa=False,
            massa_rotulada_csv_path=rdir / "total-rotulado.csv",
            sessions_csv_path=rdir / "sessions.csv",
            modo="gerar",
            defaults={"num_traces": 20, "seed": 42, "discovery_mode": "petri"},
            events_params=copy.deepcopy(selecionados),
        )
        res = gerar_eventos(ev_cfg, root=ROOT, work_dir=rdir / "_work")
        saidas.append(Path(res["output_events_csv_path"]).read_bytes())
    assert saidas[0] == saidas[1]

    # Playout e rotulagem por padrao direto na CLI local (tools/pm4py_cli/run.py)
    cli = [sys.executable, str(ROOT / "tools" / "pm4py_cli" / "run.py")]
    seed_csv = ROOT / raw["events"]["events_params"][0]["seed_csv_path"]
    pnml_dir = tmp_path / "cli_disc"
    subprocess.run(
        cli + ["discovery", "--mode", "petri", "--algo", "heuristics", "--csv", str(seed_csv), "--output_dir", str(pnml_dir), "--yes"],
        check=True,
        capture_output=True,
    )
    pnml = pnml_dir / "discovery" / f"{seed_csv.stem}-petri.pnml"
    cli_sims = []
    for k in ("p1", "p2"):
        pdir = tmp_path / k
        subprocess.run(
            cli + ["playout", "--pnml", str(pnml), "--csv", str(seed_csv), "--num_traces", "10", "--seed", "42", "--output_dir", str(pdir), "--yes"],
            check=True,
            capture_output=True,
        )
        cli_sims.append((pdir / "playout" / f"{seed_csv.stem}-simulado.csv").read_bytes())
    assert cli_sims[0] == cli_sims[1]

    tiny_csv = tmp_path / "tiny.csv"
    pd.DataFrame(
        [
            {"cd_sessao": "S_MATCH", "cd_produto": "p", "cd_sub_produto": "s", "page_name": "A", "cd_evento_tipo": "page_view", "button_text": "", "timestamp": "2025-01-01 00:00:01"},
            {"cd_sessao": "S_MATCH", "cd_produto": "p", "cd_sub_produto": "s", "page_name": "B", "cd_evento_tipo": "page_view", "button_text": "", "timestamp": "2025-01-01 00:00:02"},
            {"cd_sessao": "S_REV", "cd_produto": "p", "cd_sub_produto": "s", "page_name": "B", "cd_evento_tipo": "page_view", "button_text": "", "timestamp": "2025-01-01 00:00:01"},
            {"cd_sessao": "S_REV", "cd_produto": "p", "cd_sub_produto": "s", "page_name": "A", "cd_evento_tipo": "page_view", "button_text": "", "timestamp": "2025-01-01 00:00:02"},
            {"cd_sessao": "S_ONLY_A", "cd_produto": "p", "cd_sub_produto": "s", "page_name": "A", "cd_evento_tipo": "page_view", "button_text": "", "timestamp": "2025-01-01 00:00:01"},
        ]
    ).to_csv(tiny_csv, index=False)
    out_tiny = tmp_path / "tiny_rot.csv"
    subprocess.run(
        cli + [
            "labeling",
            "--csv", str(tiny_csv),
            "--pattern", "[p:s:A].page_view()", "[p:s:B].page_view()",
            "--label_col", "is_x",
            "--output_csv", str(out_tiny),
            "--yes",
        ],
        check=True,
        capture_output=True,
    )
    df_tiny = pd.read_csv(out_tiny).groupby("cd_sessao")["is_x"].max()
    assert int(df_tiny["S_MATCH"]) == 1
    assert int(df_tiny["S_REV"]) == 0
    assert int(df_tiny["S_ONLY_A"]) == 0
