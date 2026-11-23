"""Gerador e rotulador de massas sintéticas guiado pela configuração JSON da versão."""
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experience_mining.configuracao import ConfigEventos, ConfigRotulagem, ErroConfiguracao  # noqa: E402
from experience_mining.nucleo.rotulo import (  # noqa: E402
    classificar_experiencia,
    extract_primary_label,
    mapas_absorcao_de_regras,
)

DEFAULTS_GERACAO = {
    "num_traces": 100,
    "seed": 42,
    "discovery_algo": "heuristics",
    "discovery_mode": "both",
    "min_events": None,
    "start_session_at": 1,
}


def resolve_pm4py_cmd(root: Path | str | None = None) -> list[str]:
    """Resolve o pm4py-cli preferindo a cópia local em tools/pm4py_cli/run.py."""
    raiz = Path(root).resolve() if root is not None else ROOT_DIR
    tools_run = raiz / "tools" / "pm4py_cli" / "run.py"
    if tools_run.exists():
        return [sys.executable, str(tools_run)]

    cli_exe = shutil.which("pm4py-cli") or shutil.which("pm4py-cli.exe")
    if cli_exe:
        return [cli_exe]

    cli_bat = shutil.which("pm4py-cli.bat")
    if cli_bat:
        return [cli_bat]

    return [sys.executable, "-m", "jornada_ux.main"]


def _executar_pm4py(cmd_args: list[str], cwd: Path | None = None) -> None:
    cmd_str = " ".join(f'"{a}"' if (" " in str(a) or "[" in str(a)) else str(a) for a in cmd_args)
    print(f"[EXEC] {cmd_str}")
    res = subprocess.run(cmd_args, cwd=str(cwd) if cwd else None)
    if res.returncode != 0:
        raise RuntimeError(f"Erro ao executar pm4py-cli (codigo {res.returncode}): {cmd_str}")


def _importar_pm4py_cli_modulos(root: Path):
    """Garante importação do pacote jornada_ux a partir de tools/pm4py_cli/src do repositório."""
    cli_src = str((Path(root).resolve() / "tools" / "pm4py_cli" / "src"))
    if cli_src in sys.path:
        sys.path.remove(cli_src)
    sys.path.insert(0, cli_src)

    mod_existente = sys.modules.get("jornada_ux")
    if mod_existente is not None:
        arq = getattr(mod_existente, "__file__", "") or ""
        if Path(cli_src).resolve() not in Path(arq).resolve().parents:
            for nome in [k for k in list(sys.modules) if k == "jornada_ux" or k.startswith("jornada_ux.")]:
                del sys.modules[nome]

    from jornada_ux.modules.analysis.filter import apply_all_filters
    from jornada_ux.modules.utils import prepare_event_log

    return prepare_event_log, apply_all_filters


def resolver_parametros_cenarios(secao: ConfigEventos) -> list[dict]:
    """Combina defaults da seção `events` com sobrescritas individuais de `events_params`."""
    base = {**DEFAULTS_GERACAO, **dict(secao.defaults or {})}
    resolvidos = []
    for item in secao.events_params:
        merged = {**base, **dict(item)}
        resolvidos.append(merged)
    return resolvidos


def _validar_sementes(cenarios: list[dict], raiz: Path) -> list[tuple[dict, Path]]:
    validados: list[tuple[dict, Path]] = []
    for sc in cenarios:
        cid = str(sc["id"])
        raw_seed = Path(sc["seed_csv_path"])
        seed_path = raw_seed if raw_seed.is_absolute() else (raiz / raw_seed)
        if not seed_path.exists() or not seed_path.is_file() or seed_path.suffix.lower() != ".csv":
            raise ErroConfiguracao(
                f"Cenário '{cid}': arquivo semente não encontrado (seed_csv_path): {sc['seed_csv_path']}"
            )
        if sc["discovery_mode"] == "dfg":
            raise ErroConfiguracao(
                f"Cenário '{cid}': discovery_mode='dfg' não gera .pnml para o playout; use 'petri' ou 'both'."
            )
        validados.append((sc, seed_path))
    return validados


def gerar_eventos(
    secao: ConfigEventos,
    root: Path | str = ROOT_DIR,
    work_dir: Path | str | None = None,
) -> dict:
    """Gera a massa sintética por cenário a partir de ConfigEventos validada."""
    if not secao.events_params:
        raise ErroConfiguracao("events.events_params não pode ser vazio no modo geração.")
    if secao.output_events_csv_path is None:
        raise ErroConfiguracao("output_events_csv_path obrigatório no modo geração.")

    raiz = Path(root).resolve()
    cenarios = resolver_parametros_cenarios(secao)
    cenarios_validados = _validar_sementes(cenarios, raiz)

    out_csv = Path(secao.output_events_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(work_dir) if work_dir is not None else (out_csv.parent / "_work")
    workdir.mkdir(parents=True, exist_ok=True)

    pm4py_base_cmd = resolve_pm4py_cmd(raiz)
    current_session = int((secao.defaults or {}).get("start_session_at", DEFAULTS_GERACAO["start_session_at"]))

    dfs: list[pd.DataFrame] = []
    relatorio_cenarios: list[dict] = []

    print("=" * 80)
    print("GERACAO DE MASSA SINTETICA POR CENARIO")
    print(f"Executavel pm4py-cli: {' '.join(pm4py_base_cmd)}")
    print(f"Saida consolidada: {out_csv}")
    print("=" * 80)

    for sc, seed_path in cenarios_validados:
        cid = str(sc["id"])
        num_traces = int(sc["num_traces"])
        seed = int(sc["seed"])
        algo = str(sc["discovery_algo"])
        mode = str(sc["discovery_mode"])
        min_events = sc.get("min_events")

        scen_dir = workdir / "cenarios" / cid
        scen_dir.mkdir(parents=True, exist_ok=True)
        pnml_path = scen_dir / "discovery" / f"{seed_path.stem}-petri.pnml"
        simulated_csv = scen_dir / "playout" / f"{seed_path.stem}-simulado.csv"

        print(f"\n>>> Cenario [{cid}] | semente={seed_path.name} | traces={num_traces} | seed={seed}")
        _executar_pm4py(
            pm4py_base_cmd
            + [
                "discovery",
                "--mode",
                mode,
                "--algo",
                algo,
                "--csv",
                str(seed_path),
                "--output_dir",
                str(scen_dir),
                "--yes",
            ],
            cwd=raiz,
        )
        if not pnml_path.exists():
            raise RuntimeError(f"Discovery do cenário '{cid}' não gerou o arquivo PNML esperado: {pnml_path}")

        _executar_pm4py(
            pm4py_base_cmd
            + [
                "playout",
                "--pnml",
                str(pnml_path),
                "--csv",
                str(seed_path),
                "--start_session_at",
                str(current_session),
                "--num_traces",
                str(num_traces),
                "--seed",
                str(seed),
                "--output_dir",
                str(scen_dir),
                "--yes",
            ],
            cwd=raiz,
        )
        if not simulated_csv.exists():
            raise RuntimeError(f"Playout do cenário '{cid}' não gerou o arquivo simulado esperado: {simulated_csv}")

        current_session += num_traces

        df_sim = pd.read_csv(simulated_csv)
        sessoes_simuladas = int(df_sim["cd_sessao"].nunique()) if len(df_sim) > 0 else 0
        if min_events is not None and len(df_sim) > 0:
            counts = df_sim.groupby("cd_sessao")["cd_sessao"].transform("count")
            df_filtrado = df_sim[counts >= int(min_events)].copy()
        else:
            df_filtrado = df_sim.copy()

        sessoes_finais = int(df_filtrado["cd_sessao"].nunique()) if len(df_filtrado) > 0 else 0
        removidas = sessoes_simuladas - sessoes_finais
        eventos_finais = int(len(df_filtrado))

        if len(df_filtrado) > 0:
            dfs.append(df_filtrado)

        relatorio_cenarios.append(
            {
                "cenario_id": cid,
                "seed_csv_path": str(sc["seed_csv_path"]),
                "num_traces": num_traces,
                "seed": seed,
                "discovery_algo": algo,
                "discovery_mode": mode,
                "min_events": min_events if min_events is not None else "",
                "sessoes_simuladas": sessoes_simuladas,
                "removidas_min_events": removidas,
                "sessoes_finais": sessoes_finais,
                "eventos_finais": eventos_finais,
            }
        )

    if not dfs:
        raise RuntimeError("Nenhuma sessão restou após geração e filtro min_events.")

    df_total = pd.concat(dfs, ignore_index=True)
    df_total.to_csv(out_csv, index=False)

    df_rel = pd.DataFrame(relatorio_cenarios)
    relatorio_csv = out_csv.parent / "relatorio_geracao.csv"
    df_rel.to_csv(relatorio_csv, index=False)

    totais = {
        "sessoes_simuladas": int(df_rel["sessoes_simuladas"].sum()),
        "removidas_min_events": int(df_rel["removidas_min_events"].sum()),
        "sessoes_finais": int(df_rel["sessoes_finais"].sum()),
        "eventos_finais": int(df_rel["eventos_finais"].sum()),
    }

    print("\n" + "-" * 80)
    print("RELATORIO DA GERACAO POR CENARIO")
    print("-" * 80)
    print(df_rel[["cenario_id", "num_traces", "min_events", "sessoes_simuladas", "removidas_min_events", "sessoes_finais", "eventos_finais"]].to_string(index=False))
    print("-" * 80)
    print(
        f"TOTAIS | simuladas={totais['sessoes_simuladas']} "
        f"| removidas_min_events={totais['removidas_min_events']} "
        f"| finais={totais['sessoes_finais']} "
        f"| eventos={totais['eventos_finais']}"
    )
    print(f"[+] Massa consolidada salva em: {out_csv}")
    print(f"[+] Relatorio salvo em: {relatorio_csv}")

    return {
        "output_events_csv_path": out_csv,
        "relatorio_csv_path": relatorio_csv,
        "cenarios": relatorio_cenarios,
        "totais": totais,
    }


def rotular_massa(
    secao: ConfigRotulagem,
    events_csv_path: Path | str,
    root: Path | str = ROOT_DIR,
    output_csv_override: Path | str | None = None,
) -> Path:
    """Aplica todas as regras de labeling_params sobre a massa inteira e grava primary_label e experience."""
    if not secao.labeling_params:
        raise ErroConfiguracao("labeling.labeling_params não pode ser vazio no modo geração.")

    ev_path = Path(events_csv_path)
    if not ev_path.exists():
        raise ErroConfiguracao(f"Arquivo de eventos para rotulagem não encontrado: {ev_path}")

    destino = Path(output_csv_override) if output_csv_override is not None else secao.output_labeling_csv_path
    if destino is None:
        raise ErroConfiguracao("output_labeling_csv_path obrigatório no modo geração.")

    raiz = Path(root).resolve()
    prepare_event_log, apply_all_filters = _importar_pm4py_cli_modulos(raiz)

    print(f"[*] Rotulando massa inteira: {ev_path}")
    df_raw, _, df_event_log = prepare_event_log(str(ev_path), ["page_view", "click_button"], temp_col=True)
    if df_event_log is None or df_raw is None:
        raise RuntimeError(f"Falha ao preparar event log para rotulagem: {ev_path}")

    session_ids = list(dict.fromkeys(df_raw["cd_sessao"].tolist()))
    df_sess = pd.DataFrame({"cd_sessao": session_ids})

    is_cols: list[str] = []
    for lbl in secao.labeling_params:
        col = lbl["label_col"]
        is_cols.append(col)
        ns = SimpleNamespace(
            contain=lbl.get("contain"),
            exclude=lbl.get("exclude"),
            pattern=lbl.get("pattern"),
            not_pattern=lbl.get("not_pattern"),
            start_activities=None,
            end_activities=None,
            trace_include=None,
            trace_exclude=None,
            min_duration=None,
            max_duration=None,
            min_events=None,
            max_events=None,
        )
        df_filtrado = apply_all_filters(df_event_log, ns)
        matching_ids = (
            set(df_filtrado["case:concept:name"].unique())
            if df_filtrado is not None and len(df_filtrado) > 0
            else set()
        )
        df_sess[col] = df_sess["cd_sessao"].isin(matching_ids).astype(int)

    compostas, abandono_compostas = mapas_absorcao_de_regras(secao.labeling_params)
    df_sess["primary_label"] = df_sess[is_cols].apply(
        lambda r: extract_primary_label(r, compostas=compostas, abandono_compostas=abandono_compostas),
        axis=1,
    )
    df_sess["experience"] = df_sess[is_cols].apply(
        lambda r: classificar_experiencia({c: int(r[c]) for c in is_cols}),
        axis=1,
    )

    colunas_saida = ["cd_sessao", "primary_label", "experience", *is_cols]
    df_sess = df_sess[colunas_saida]
    destino.parent.mkdir(parents=True, exist_ok=True)
    df_sess.to_csv(destino, index=False)
    print(f"[+] Massa rotulada ({len(df_sess)} sessoes, {len(is_cols)} flags) salva em: {destino}")
    return destino


if __name__ == "__main__":
    sys.exit(
        "dataset/gerar_massas.py nao aceita argumentos CLI. "
        "Execute o pipeline via `python run_all.py --config <arquivo.json>`."
    )
