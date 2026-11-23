"""Configurações tipadas, carregamento e validação de JSON por versão."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
import jsonschema

from experience_mining.io import (
    carregar_agrupamento,
    carregar_representacao,
    ler_json,
    validar_contrato_eventos_csv,
    validar_contrato_rotulo_csv,
    validar_formato_chave,
)

REP_JSON_PARA_MODELO = {"tfidf": "tf_idf", "minilm": "minilm", "gemini": "gemini"}
MODELO_PARA_REP_JSON = {v: k for k, v in REP_JSON_PARA_MODELO.items()}

TRADUCAO_TIPOS = {
    "object": "objeto",
    "array": "lista",
    "string": "texto",
    "integer": "inteiro",
    "boolean": "booleano",
}


class ErroConfiguracao(ValueError):
    """Erro de esquema, seleção, formato ou contrato antes da execução do pipeline."""


@dataclass
class ConfigEventos:
    events_csv_path: Path
    output_events_csv_path: Path | None
    rotular_massa: bool
    massa_rotulada_csv_path: Path
    sessions_csv_path: Path
    incluir_produto: bool = True
    modo: str = "carregar"
    defaults: dict = field(default_factory=dict)
    events_params: list[dict] = field(default_factory=list)


@dataclass
class ConfigRotulagem:
    modo: str = "carregar"
    labeling_csv_path: Path | None = None
    output_labeling_csv_path: Path | None = None
    labeling_params: list[dict] = field(default_factory=list)


@dataclass
class ConfigRepresentacaoItem:
    nome: str
    modo: str
    path: Path
    params: dict = field(default_factory=dict)


@dataclass
class ConfigRepresentacoes:
    gerar: bool
    tfidf_path: Path
    minilm_path: Path
    gemini_path: Path
    minilm_model_name: str = "all-MiniLM-L6-v2"
    gemini_model_name: str = "models/gemini-embedding-001"
    batch_size: int = 32
    max_features: int | None = None
    min_df: int = 1
    itens: dict[str, ConfigRepresentacaoItem] = field(default_factory=dict)

    def caminho_por_sufixo(self, sufixo: str) -> Path:
        mapa = {
            "tf_idf": self.tfidf_path,
            "tfidf": self.tfidf_path,
            "minilm": self.minilm_path,
            "gemini": self.gemini_path,
        }
        return mapa[sufixo]


@dataclass
class ConfigAgrupadorItem:
    algoritmo: str
    modo: str
    caminhos_por_rep: dict[str, Path] = field(default_factory=dict)
    output_clusters_dir: Path | None = None
    params: dict = field(default_factory=dict)


@dataclass
class ConfigAgrupamentos:
    gerar: bool
    clusters_dir: Path
    auto_k: bool = True
    n_clusters: int = 7
    kmeans_seed: int = 42
    auto_grid: bool = True
    min_cluster_size: int = 5
    min_samples: int | None = None
    hdbscan_metric: str = "euclidean"
    itens: dict[str, ConfigAgrupadorItem] = field(default_factory=dict)
    caminhos_modelo: dict[str, Path] = field(default_factory=dict)

    def caminho_modelo(self, nome_modelo: str) -> Path:
        if nome_modelo in self.caminhos_modelo:
            return self.caminhos_modelo[nome_modelo]
        return Path(self.clusters_dir) / f"{nome_modelo}.pkl"

    def modelos_ativos(self, ordem_padrao: list[str]) -> list[str]:
        if not self.caminhos_modelo:
            return [m for m in ordem_padrao if (Path(self.clusters_dir) / f"{m}.pkl").exists()]
        return [m for m in ordem_padrao if m in self.caminhos_modelo]


@dataclass
class ConfigMetricas:
    enabled: bool
    output_metrics_dir: Path
    output_dados_dir: Path
    summary_csv_path: Path
    include: list[str] | None = None


@dataclass
class ConfigShap:
    enabled: bool
    output_jornadas_dir: Path
    output_figures_dir: Path
    output_dados_dir: Path
    random_state: int = 42
    revised_titles: bool = False
    gerar_individuais: bool = True
    surrogate: str = "xgboost"
    colsample_sensitivity: dict | None = None


@dataclass
class ConfigJornadasGrupo:
    enabled: bool
    output_jornadas_dir: Path
    tsne_seed: int = 42


@dataclass
class ConfigPm4py:
    enabled: bool
    output_jornadas_dir: Path
    dfg: bool = True
    pnml: bool = True


@dataclass
class ConfigFiguras:
    enabled: bool
    output_figures_dir: Path
    tsne_seed: int = 42
    include: list[str] | None = None
    revised_titles: bool = False
    highlight_model: str | None = "hdbscan_tf_idf"
    shap_panel_clusters: list[int] | None = None


@dataclass
class ConfigPipeline:
    root: Path
    output_dir: Path
    eventos: ConfigEventos
    representacoes: ConfigRepresentacoes
    agrupamentos: ConfigAgrupamentos
    metricas: ConfigMetricas
    shap: ConfigShap
    jornadas_grupo: ConfigJornadasGrupo
    pm4py: ConfigPm4py
    figuras: ConfigFiguras
    name: str = "v1.1"
    labeling: ConfigRotulagem | None = None


def resolver_caminho_config(cli_config: str | None, root: Path | str = ".") -> Path:
    """Resolve a configuração na ordem: --config > EM_CONFIG (ambiente ou .env) > erro."""
    raiz = Path(root).resolve()
    env_path = raiz / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    if cli_config:
        p = Path(cli_config)
        p_res = p if p.is_absolute() else (raiz / p if (raiz / p).exists() else p)
        if not p_res.exists():
            raise ErroConfiguracao(f"Arquivo de configuração não encontrado: {cli_config}")
        return p_res.resolve()

    env_cfg = os.environ.get("EM_CONFIG", "").strip()
    if not env_cfg:
        raise ErroConfiguracao(
            "Nenhuma configuração informada. Use `--config <arquivo.json>` ou defina `EM_CONFIG` "
            "no ambiente / arquivo `.env` (veja o modelo em `.env.example`)."
        )
    p = Path(env_cfg)
    p_res = p if p.is_absolute() else (raiz / p if (raiz / p).exists() else p)
    if not p_res.exists():
        raise ErroConfiguracao(f"Arquivo indicado em EM_CONFIG não encontrado: {env_cfg}")
    return p_res.resolve()


def _resolver_caminho(root: Path, valor: str) -> Path:
    p = Path(valor)
    return p if p.is_absolute() else (root / p)


def _diagnosticar_modo(
    passo: str,
    obj: object,
    param_key: str,
    load_keys: tuple[str, ...],
    companion_keys: tuple[str, ...],
) -> None:
    if not isinstance(obj, dict):
        return
    tem_param = param_key in obj
    load_presentes = [k for k in load_keys if k in obj]
    comp_presentes = [k for k in companion_keys if k in obj]

    if tem_param and load_presentes:
        raise ErroConfiguracao(
            f"{passo}: declare apenas gerar ('{param_key}') ou carregar ({load_presentes[0]}), nunca os dois."
        )
    if not tem_param and not load_presentes:
        raise ErroConfiguracao(
            f"{passo}: caminho de entrada obrigatório quando não há geração ('{param_key}')."
        )
    if comp_presentes and not tem_param:
        raise ErroConfiguracao(
            f"{passo}: {comp_presentes} só pode ser usado junto com '{param_key}'."
        )


def _traduzir_erro_schema(err: jsonschema.ValidationError) -> str:
    caminho = ".".join(str(p) for p in err.path) or "raiz"
    if err.validator == "type":
        esperado = err.validator_value
        if isinstance(esperado, list):
            esp_txt = " ou ".join(TRADUCAO_TIPOS.get(x, x) for x in esperado)
        else:
            esp_txt = TRADUCAO_TIPOS.get(esperado, str(esperado))
        return f"{caminho} deve ser {esp_txt}"
    if err.validator == "enum":
        aceitos = ", ".join(repr(x) for x in err.validator_value)
        return f"{caminho}: valor inválido {err.instance!r}; aceitos: {aceitos}"
    if err.validator == "additionalProperties" and isinstance(err.instance, dict):
        permitidas = set(err.schema.get("properties", {}))
        extras = sorted(set(err.instance) - permitidas)
        return f"Chave desconhecida em {caminho}: {', '.join(extras)}"
    if err.validator == "required":
        return f"{caminho}: {err.message}"
    return f"{caminho}: {err.message}"


def _validar_esquema(raw: dict, schema_path: Path) -> None:
    if not isinstance(raw, dict):
        raise ErroConfiguracao("A raiz da configuração deve ser um objeto JSON.")

    _diagnosticar_modo("events", raw.get("events"), "events_params", ("events_csv_path",), ("defaults", "output_events_csv_path"))
    _diagnosticar_modo("labeling", raw.get("labeling"), "labeling_params", ("labeling_csv_path",), ("output_labeling_csv_path",))

    reps = raw.get("representations")
    if isinstance(reps, dict):
        for rep in ("tfidf", "minilm", "gemini"):
            if rep in reps:
                _diagnosticar_modo(
                    f"representations.{rep}",
                    reps[rep],
                    f"{rep}_params",
                    (f"{rep}_pkl_path", f"{rep}_dir_path", f"{rep}_npy_path"),
                    (f"output_{rep}_dir_path",),
                )

    clusters = raw.get("clusters")
    if isinstance(clusters, dict):
        load_keys = tuple(
            f"{r}_{ext}_path" for r in ("tfidf", "minilm", "gemini") for ext in ("pkl", "dir", "npy")
        )
        for algo in ("kmeans", "hdbscan"):
            if algo in clusters:
                _diagnosticar_modo(
                    f"clusters.{algo}",
                    clusters[algo],
                    f"{algo}_params",
                    load_keys,
                    ("output_clusters_dir_path",),
                )

    schema = ler_json(schema_path)
    validador = jsonschema.Draft202012Validator(schema)
    erros = sorted(validador.iter_errors(raw), key=lambda e: list(e.path))
    if erros:
        primeiro = erros[0]
        if primeiro.validator == "oneOf" and primeiro.context:
            sub = min(primeiro.context, key=lambda e: len(e.path))
            raise ErroConfiguracao(_traduzir_erro_schema(sub))
        raise ErroConfiguracao(_traduzir_erro_schema(primeiro))


def carregar_config_json(
    config_path: Path | str,
    root: Path | str = ".",
    validar_contratos: bool = True,
) -> ConfigPipeline:
    """Lê, valida esquema/arquivos/contratos e monta o ConfigPipeline para o ponto de entrada."""
    raiz = Path(root).resolve()
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = (raiz / cfg_file) if (raiz / cfg_file).exists() else cfg_file
    if not cfg_file.exists():
        raise ErroConfiguracao(f"Arquivo de configuração não encontrado: {config_path}")

    raw = ler_json(cfg_file)
    _validar_esquema(raw, raiz / "configs" / "schema.json")

    name = raw.get("name") or cfg_file.stem
    saida = _resolver_caminho(raiz, raw["output_dir_path"]) if "output_dir_path" in raw else (raiz / "results" / name)
    metrics_dir = saida / "metrics"
    dados_dir = saida / "dados"
    figures_dir = saida / "figures"
    jornadas_dir = saida / "jornadas_por_grupo"
    sessions_csv = raiz / "data" / name / "sessions_represented.csv"

    # Eventos
    ev_raw = raw["events"]
    event_ids: list | None = None
    if "events_csv_path" in ev_raw:
        ev_csv = _resolver_caminho(raiz, ev_raw["events_csv_path"])
        if validar_contratos:
            try:
                validar_formato_chave("events.events_csv_path", ev_csv)
                event_ids = validar_contrato_eventos_csv(ev_csv)
            except ValueError as exc:
                raise ErroConfiguracao(str(exc)) from exc
        cfg_ev = ConfigEventos(
            events_csv_path=ev_csv,
            output_events_csv_path=None,
            rotular_massa=False,
            massa_rotulada_csv_path=ev_csv,
            sessions_csv_path=sessions_csv,
            modo="carregar",
        )
    else:
        out_ev = (
            _resolver_caminho(raiz, ev_raw["output_events_csv_path"])
            if "output_events_csv_path" in ev_raw
            else raiz / "dataset" / "processed" / name / "total.csv"
        )
        if validar_contratos:
            for sc in ev_raw["events_params"]:
                seed_p = _resolver_caminho(raiz, sc["seed_csv_path"])
                try:
                    validar_formato_chave(f"events.events_params[{sc['id']}].seed_csv_path", seed_p)
                except ValueError as exc:
                    raise ErroConfiguracao(str(exc)) from exc
        cfg_ev = ConfigEventos(
            events_csv_path=out_ev,
            output_events_csv_path=out_ev,
            rotular_massa=False,
            massa_rotulada_csv_path=out_ev,
            sessions_csv_path=sessions_csv,
            modo="gerar",
            defaults=dict(ev_raw.get("defaults", {})),
            events_params=list(ev_raw["events_params"]),
        )

    # Rotulagem
    lbl_raw = raw["labeling"]
    if "labeling_csv_path" in lbl_raw:
        lbl_csv = _resolver_caminho(raiz, lbl_raw["labeling_csv_path"])
        if validar_contratos:
            try:
                validar_formato_chave("labeling.labeling_csv_path", lbl_csv)
                validar_contrato_rotulo_csv(lbl_csv, event_session_ids=event_ids)
            except ValueError as exc:
                raise ErroConfiguracao(str(exc)) from exc
        cfg_lbl = ConfigRotulagem(modo="carregar", labeling_csv_path=lbl_csv)
        cfg_ev.massa_rotulada_csv_path = lbl_csv
    else:
        out_lbl = (
            _resolver_caminho(raiz, lbl_raw["output_labeling_csv_path"])
            if "output_labeling_csv_path" in lbl_raw
            else raiz / "dataset" / "processed" / name / "total-rotulado.csv"
        )
        cfg_lbl = ConfigRotulagem(
            modo="gerar",
            output_labeling_csv_path=out_lbl,
            labeling_params=list(lbl_raw["labeling_params"]),
        )
        cfg_ev.massa_rotulada_csv_path = out_lbl

    # Representações
    rep_itens: dict[str, ConfigRepresentacaoItem] = {}
    rep_ids_por_nome: dict[str, list] = {}
    for rep_nome, rep_sec in raw["representations"].items():
        param_key = f"{rep_nome}_params"
        if param_key in rep_sec:
            out_rep = (
                _resolver_caminho(raiz, rep_sec[f"output_{rep_nome}_dir_path"])
                if f"output_{rep_nome}_dir_path" in rep_sec
                else raiz / "data" / name / rep_nome
            )
            rep_itens[rep_nome] = ConfigRepresentacaoItem(
                nome=rep_nome,
                modo="gerar",
                path=out_rep,
                params=dict(rep_sec[param_key]),
            )
        else:
            chave = next(iter(rep_sec))
            rep_path = _resolver_caminho(raiz, rep_sec[chave])
            if validar_contratos:
                try:
                    validar_formato_chave(f"representations.{rep_nome}.{chave}", rep_path)
                    rep_data = carregar_representacao(rep_path, expected_session_ids=event_ids)
                    rep_ids_por_nome[rep_nome] = list(rep_data["session_ids"])
                except ValueError as exc:
                    raise ErroConfiguracao(str(exc)) from exc
            rep_itens[rep_nome] = ConfigRepresentacaoItem(nome=rep_nome, modo="carregar", path=rep_path)

    tfidf_item = rep_itens.get("tfidf")
    minilm_item = rep_itens.get("minilm")
    gemini_item = rep_itens.get("gemini")
    cfg_reps = ConfigRepresentacoes(
        gerar=any(item.modo == "gerar" for item in rep_itens.values()),
        tfidf_path=tfidf_item.path if tfidf_item else raiz / "data" / name / "tfidf",
        minilm_path=minilm_item.path if minilm_item else raiz / "data" / name / "minilm",
        gemini_path=gemini_item.path if gemini_item else raiz / "data" / name / "gemini",
        minilm_model_name=(minilm_item.params.get("model_name", "all-MiniLM-L6-v2") if minilm_item else "all-MiniLM-L6-v2"),
        gemini_model_name=(gemini_item.params.get("model_name", "models/gemini-embedding-001") if gemini_item else "models/gemini-embedding-001"),
        batch_size=(minilm_item.params.get("batch_size", 32) if minilm_item else 32),
        max_features=(tfidf_item.params.get("max_features") if tfidf_item else None),
        min_df=(tfidf_item.params.get("min_df", 1) if tfidf_item else 1),
        itens=rep_itens,
    )

    # Agrupamentos
    cl_itens: dict[str, ConfigAgrupadorItem] = {}
    caminhos_modelo: dict[str, Path] = {}
    clusters_dir_default = raiz / "clusters" / name
    for algo, cl_sec in raw["clusters"].items():
        param_key = f"{algo}_params"
        if param_key in cl_sec:
            out_cl = (
                _resolver_caminho(raiz, cl_sec["output_clusters_dir_path"])
                if "output_clusters_dir_path" in cl_sec
                else clusters_dir_default
            )
            caminhos_por_rep = {}
            for rep_nome in rep_itens:
                mod_rep = REP_JSON_PARA_MODELO[rep_nome]
                destino = out_cl / f"{algo}_{mod_rep}"
                caminhos_por_rep[mod_rep] = destino
                caminhos_modelo[f"{algo}_{mod_rep}"] = destino
            cl_itens[algo] = ConfigAgrupadorItem(
                algoritmo=algo,
                modo="gerar",
                caminhos_por_rep=caminhos_por_rep,
                output_clusters_dir=out_cl,
                params=dict(cl_sec[param_key]),
            )
        else:
            caminhos_por_rep = {}
            for chave, valor in cl_sec.items():
                rep_nome = chave.split("_", 1)[0]
                mod_rep = REP_JSON_PARA_MODELO[rep_nome]
                cl_path = _resolver_caminho(raiz, valor)
                if validar_contratos:
                    expected_ids = rep_ids_por_nome.get(rep_nome) or event_ids
                    try:
                        validar_formato_chave(f"clusters.{algo}.{chave}", cl_path)
                        carregar_agrupamento(cl_path, expected_session_ids=expected_ids)
                    except ValueError as exc:
                        raise ErroConfiguracao(str(exc)) from exc
                caminhos_por_rep[mod_rep] = cl_path
                caminhos_modelo[f"{algo}_{mod_rep}"] = cl_path
            cl_itens[algo] = ConfigAgrupadorItem(algoritmo=algo, modo="carregar", caminhos_por_rep=caminhos_por_rep)

    km_params = cl_itens["kmeans"].params if "kmeans" in cl_itens else {}
    hd_params = cl_itens["hdbscan"].params if "hdbscan" in cl_itens else {}
    cfg_clusters = ConfigAgrupamentos(
        gerar=any(item.modo == "gerar" for item in cl_itens.values()),
        clusters_dir=clusters_dir_default,
        auto_k=km_params.get("auto_k", True),
        n_clusters=km_params.get("n_clusters", 7),
        kmeans_seed=km_params.get("seed", 42),
        auto_grid=hd_params.get("auto_grid", True),
        min_cluster_size=hd_params.get("min_cluster_size", 5),
        min_samples=hd_params.get("min_samples", None),
        hdbscan_metric=hd_params.get("metric", "euclidean"),
        itens=cl_itens,
        caminhos_modelo=caminhos_modelo,
    )

    # Passos de resultado (padrão = executar tudo)
    met_raw = raw.get("metrics", {})
    cfg_met = ConfigMetricas(
        enabled=met_raw.get("enabled", True),
        output_metrics_dir=metrics_dir,
        output_dados_dir=dados_dir,
        summary_csv_path=saida / "summary_metrics.csv",
        include=met_raw.get("include"),
    )

    fig_raw = raw.get("figures", {})
    fig_enabled = fig_raw.get("enabled", True)
    fig_include = fig_raw.get("include")
    revised_titles = fig_raw.get("revised_titles", False)
    highlight_model = fig_raw["highlight_model"] if "highlight_model" in fig_raw else "hdbscan_tf_idf"
    shap_panel_clusters = fig_raw.get("shap_panel_clusters")

    shap_raw = raw.get("shap", {})
    cfg_shap = ConfigShap(
        enabled=shap_raw.get("enabled", True),
        output_jornadas_dir=jornadas_dir,
        output_figures_dir=figures_dir,
        output_dados_dir=dados_dir,
        random_state=shap_raw.get("seed", 42),
        revised_titles=revised_titles,
        gerar_individuais=fig_enabled and (fig_include is None or "beeswarm" in fig_include),
        surrogate=shap_raw.get("surrogate", "xgboost"),
        colsample_sensitivity=shap_raw.get("colsample_sensitivity"),
    )

    jg_raw = raw.get("journeys_by_cluster", {})
    cfg_jg = ConfigJornadasGrupo(
        enabled=jg_raw.get("enabled", True),
        output_jornadas_dir=jornadas_dir,
        tsne_seed=jg_raw.get("tsne_seed", 42),
    )

    pm_raw = raw.get("pm4py", {})
    pm_dfg = pm_raw.get("dfg", True)
    pm_pnml = pm_raw.get("pnml", True)
    cfg_pm4py = ConfigPm4py(
        enabled=bool(pm_dfg or pm_pnml),
        output_jornadas_dir=jornadas_dir,
        dfg=pm_dfg,
        pnml=pm_pnml,
    )

    cfg_fig = ConfigFiguras(
        enabled=fig_enabled,
        output_figures_dir=figures_dir,
        tsne_seed=jg_raw.get("tsne_seed", 42),
        include=fig_include,
        revised_titles=revised_titles,
        highlight_model=highlight_model,
        shap_panel_clusters=shap_panel_clusters,
    )

    return ConfigPipeline(
        root=raiz,
        output_dir=saida,
        eventos=cfg_ev,
        representacoes=cfg_reps,
        agrupamentos=cfg_clusters,
        metricas=cfg_met,
        shap=cfg_shap,
        jornadas_grupo=cfg_jg,
        pm4py=cfg_pm4py,
        figuras=cfg_fig,
        name=name,
        labeling=cfg_lbl,
    )


def config_padrao_v1_1(
    root: Path | str = ".",
    output_dir: Path | str | None = None,
    sessions_csv_path: Path | str | None = None,
    reusar_agrupamentos: bool = False,
    rotular_massa: bool = False,
    sem_shap: bool = False,
    petri: bool = False,
    revised_titles: bool = False,
) -> ConfigPipeline:
    """Monta a configuração padrão equivalente ao pipeline v1.1."""
    raiz = Path(root).resolve()
    saida = Path(output_dir).resolve() if output_dir is not None else raiz / "results"
    sessoes_csv = (
        Path(sessions_csv_path).resolve()
        if sessions_csv_path is not None
        else (
            (raiz / "data" / "v1.1" / "sessions_represented.csv")
            if (raiz / "data" / "v1.1" / "sessions_represented.csv").exists()
            else (raiz / "data" / "sessions_represented.csv")
        )
    )
    metrics_dir = saida / "metrics"
    dados_dir = saida / "dados"
    figures_dir = saida / "figures"
    jornadas_dir = saida / "jornadas_por_grupo"
    data_dir = (raiz / "data" / "v1.0") if (raiz / "data" / "v1.0").exists() else (raiz / "data")
    clusters_dir = (raiz / "clusters" / "v1.0") if (raiz / "clusters" / "v1.0").exists() else (raiz / "clusters")

    return ConfigPipeline(
        root=raiz,
        output_dir=saida,
        eventos=ConfigEventos(
            events_csv_path=raiz / "dataset" / "processed" / "total.csv",
            output_events_csv_path=None,
            rotular_massa=rotular_massa,
            massa_rotulada_csv_path=raiz / "dataset" / "processed" / "total-rotulado-v1.1.csv",
            sessions_csv_path=sessoes_csv,
            incluir_produto=True,
        ),
        representacoes=ConfigRepresentacoes(
            gerar=not reusar_agrupamentos,
            tfidf_path=data_dir / "tfidf_vectors.pkl",
            minilm_path=data_dir / "minilm_embeddings.pkl",
            gemini_path=data_dir / "gemini_embeddings.pkl",
        ),
        agrupamentos=ConfigAgrupamentos(
            gerar=not reusar_agrupamentos,
            clusters_dir=clusters_dir,
        ),
        metricas=ConfigMetricas(
            enabled=True,
            output_metrics_dir=metrics_dir,
            output_dados_dir=dados_dir,
            summary_csv_path=saida / "summary_metrics.csv",
        ),
        shap=ConfigShap(
            enabled=not sem_shap,
            output_jornadas_dir=jornadas_dir,
            output_figures_dir=figures_dir,
            output_dados_dir=dados_dir,
            revised_titles=revised_titles,
            colsample_sensitivity={
                "model": "hdbscan_tf_idf",
                "cluster": 5,
                "colsample_bytree": 0.3,
                "output_filename": "colsample_cluster5.json",
            },
        ),
        jornadas_grupo=ConfigJornadasGrupo(
            enabled=True,
            output_jornadas_dir=jornadas_dir,
        ),
        pm4py=ConfigPm4py(
            enabled=petri,
            output_jornadas_dir=jornadas_dir,
        ),
        figuras=ConfigFiguras(
            enabled=True,
            output_figures_dir=figures_dir,
            revised_titles=revised_titles,
            highlight_model="hdbscan_tf_idf",
            shap_panel_clusters=[5, 3, 10, 0],
        ),
        name="v1.1",
    )
