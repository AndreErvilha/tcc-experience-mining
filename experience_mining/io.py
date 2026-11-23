"""Leitura, validação de contrato e gravação de artefatos (CSV, JSON, .pkl legado e formato portátil)."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PTBR = {"sep": ";", "index": False, "encoding": "utf-8-sig"}

COLUNAS_EVENTOS_OBRIGATORIAS = {
    "cd_sessao",
    "timestamp",
    "cd_evento_tipo",
    "page_name",
    "button_text",
    "cd_produto",
}

REPS_PKL_V1 = {
    "tfidf": "tfidf_vectors.pkl",
    "minilm": "minilm_embeddings.pkl",
    "gemini": "gemini_embeddings.pkl",
}


def ler_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def salvar_csv(df: pd.DataFrame, path: Path | str, ptbr: bool = False, **kwargs) -> Path:
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    params = dict(CSV_PTBR) if ptbr else {"index": False}
    params.update(kwargs)
    df.to_csv(destino, **params)
    return destino


def ler_json(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def salvar_json(dados: dict, path: Path | str, ensure_ascii: bool = True) -> Path:
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=ensure_ascii)
    return destino


def validar_formato_chave(chave: str, caminho: Path | str) -> Path:
    """Confere existência e compatibilidade entre o sufixo da chave e o caminho informado."""
    p = Path(caminho)
    if not p.exists():
        raise ValueError(f"Arquivo ou pasta ausente para '{chave}': {caminho}")
    if chave.endswith("_csv_path"):
        if not p.is_file() or p.suffix.lower() != ".csv":
            raise ValueError(f"Formato divergente para '{chave}': esperado .csv, recebido {caminho}")
    elif chave.endswith("_pkl_path"):
        if not p.is_file() or p.suffix.lower() != ".pkl":
            raise ValueError(f"Formato divergente para '{chave}': esperado .pkl, recebido {caminho}")
    elif chave.endswith("_npy_path"):
        if not p.is_file() or p.suffix.lower() != ".npy":
            raise ValueError(f"Formato divergente para '{chave}': esperado .npy, recebido {caminho}")
    elif chave.endswith("_dir_path"):
        if not p.is_dir():
            raise ValueError(f"Formato divergente para '{chave}': esperado diretório, recebido {caminho}")
    return p


def validar_contrato_eventos_csv(path: Path | str) -> list:
    p = Path(path)
    df = pd.read_csv(p, usecols=lambda c: c in COLUNAS_EVENTOS_OBRIGATORIAS)
    faltantes = sorted(COLUNAS_EVENTOS_OBRIGATORIAS - set(df.columns))
    if faltantes:
        raise ValueError(f"Contrato de eventos inválido em {p}: colunas ausentes {faltantes}")
    if df["cd_sessao"].isna().any():
        raise ValueError(f"Contrato de eventos inválido em {p}: cd_sessao contém valores nulos")
    return list(dict.fromkeys(df["cd_sessao"].tolist()))


def validar_contrato_rotulo_csv(
    path: Path | str,
    event_session_ids: list | set | None = None,
    required_label_cols: list[str] | None = None,
) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    if "cd_sessao" not in df.columns:
        raise ValueError(f"Contrato de rótulo inválido em {p}: coluna 'cd_sessao' ausente")
    is_cols = [c for c in df.columns if c.startswith("is_")]
    if not is_cols:
        raise ValueError(f"Contrato de rótulo inválido em {p}: nenhuma coluna 'is_*' encontrada")
    if required_label_cols:
        faltam = [c for c in required_label_cols if c not in df.columns]
        if faltam:
            raise ValueError(f"Contrato de rótulo inválido em {p}: flags ausentes {faltam}")
    for col in ("primary_label", "experience"):
        if col in df.columns and df[col].isna().any():
            raise ValueError(f"Contrato de rótulo inválido em {p}: coluna '{col}' contém nulos")
    if event_session_ids is not None:
        ids_rot = set(df["cd_sessao"].tolist())
        ids_ev = set(event_session_ids)
        if ids_rot != ids_ev:
            raise ValueError(
                f"Contrato de rótulo inválido em {p}: sessões não coincidem com eventos "
                f"(apenas rótulo={len(ids_rot - ids_ev)}, apenas eventos={len(ids_ev - ids_rot)})"
            )
    return df


def _carregar_pkl_ou_npy(path: Path | str) -> dict:
    p = Path(path)
    if p.suffix.lower() == ".pkl":
        with open(p, "rb") as f:
            data = pickle.load(f)
    elif p.suffix.lower() == ".npy":
        data = np.load(p, allow_pickle=True).item()
    else:
        raise ValueError(f"Formato desconhecido: {p}")
    if not isinstance(data, dict):
        raise ValueError(f"Estrutura invalida em {p}: esperado dict")
    return data


def carregar_representacao_pkl(path: Path | str) -> dict:
    data = _carregar_pkl_ou_npy(path)
    for chave in ("X", "session_ids"):
        if chave not in data:
            raise ValueError(f"Contrato invalido em {path}: chave obrigatoria '{chave}' ausente")
    if len(data["X"]) != len(data["session_ids"]):
        raise ValueError(
            f"Contrato invalido em {path}: len(X)={len(data['X'])} != len(session_ids)={len(data['session_ids'])}"
        )
    return data


def carregar_agrupamento_pkl(path: Path | str) -> dict:
    data = _carregar_pkl_ou_npy(path)
    for chave in ("cluster_labels", "session_ids"):
        if chave not in data:
            raise ValueError(f"Contrato invalido em {path}: chave obrigatoria '{chave}' ausente")
    if len(data["cluster_labels"]) != len(data["session_ids"]):
        raise ValueError(
            f"Contrato invalido em {path}: len(cluster_labels)={len(data['cluster_labels'])} "
            f"!= len(session_ids)={len(data['session_ids'])}"
        )
    return data


def carregar_representacao(
    path: Path | str,
    expected_session_ids: list | set | None = None,
) -> dict:
    p = Path(path)
    data = carregar_representacao_portatil(p) if p.is_dir() else carregar_representacao_pkl(p)
    if expected_session_ids is not None:
        if set(data["session_ids"]) != set(expected_session_ids) or len(data["session_ids"]) != len(expected_session_ids):
            raise ValueError(f"Contrato inválido em {p}: ids das sessões divergem da massa")
    return data


def carregar_agrupamento(
    path: Path | str,
    expected_session_ids: list | None = None,
    X_fallback=None,
) -> dict:
    p = Path(path)
    data = carregar_agrupamento_portatil(p) if p.is_dir() else carregar_agrupamento_pkl(p)
    if expected_session_ids is not None:
        if list(data["session_ids"]) != list(expected_session_ids):
            raise ValueError(
                f"Contrato inválido em {p}: ids ou ordem das sessões divergem da representação correspondente"
            )
    if "X" not in data and X_fallback is not None:
        data = dict(data)
        data["X"] = X_fallback
    return data


def salvar_representacao_pkl(path: Path | str, data_dict: dict) -> Path:
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "wb") as f:
        pickle.dump(data_dict, f)
    return destino


def salvar_agrupamento_pkl(path: Path | str, results: dict) -> Path:
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "wb") as f:
        pickle.dump(results, f)
    return destino


def salvar_representacao_portatil(
    dir_path: Path | str,
    X: np.ndarray,
    session_ids: list,
    meta: dict | None = None,
) -> Path:
    pasta = Path(dir_path)
    pasta.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(X)
    if len(arr) != len(session_ids):
        raise ValueError("len(X) deve ser igual a len(session_ids)")
    np.save(pasta / "vectors.npy", arr)
    pd.DataFrame({"cd_sessao": list(session_ids)}).to_csv(pasta / "sessions.csv", index=False)
    meta_limpo = {k: v for k, v in (meta or {}).items() if k not in ("primary_labels", "experiences", "labels", "df")}
    salvar_json(meta_limpo, pasta / "meta.json", ensure_ascii=False)
    return pasta


def carregar_representacao_portatil(dir_path: Path | str) -> dict:
    pasta = Path(dir_path)
    for arq in ("vectors.npy", "sessions.csv", "meta.json"):
        if not (pasta / arq).exists():
            raise FileNotFoundError(f"Arquivo ausente na pasta portatil {pasta}: {arq}")
    X = np.load(pasta / "vectors.npy", allow_pickle=False)
    sess = pd.read_csv(pasta / "sessions.csv")["cd_sessao"].tolist()
    if len(X) != len(sess):
        raise ValueError(f"Contrato invalido em {pasta}: len(X)={len(X)} != len(sessions)={len(sess)}")
    meta = ler_json(pasta / "meta.json")
    return {"X": X, "session_ids": sess, **meta}


def salvar_agrupamento_portatil(
    dir_path: Path | str,
    session_ids: list,
    cluster_labels,
    params: dict | None = None,
) -> Path:
    pasta = Path(dir_path)
    pasta.mkdir(parents=True, exist_ok=True)
    labels = [int(x) for x in cluster_labels]
    if len(labels) != len(session_ids):
        raise ValueError("len(cluster_labels) deve ser igual a len(session_ids)")
    df = pd.DataFrame({"cd_sessao": list(session_ids), "grupo": labels})
    salvar_csv(df, pasta / "groups.csv", ptbr=True)
    params_limpo = {
        k: v
        for k, v in (params or {}).items()
        if k not in ("primary_labels", "experiences", "labels", "df", "X", "cluster_centers", "probabilities")
    }
    salvar_json(params_limpo, pasta / "params.json", ensure_ascii=False)
    return pasta


def carregar_agrupamento_portatil(dir_path: Path | str) -> dict:
    pasta = Path(dir_path)
    for arq in ("groups.csv", "params.json"):
        if not (pasta / arq).exists():
            raise FileNotFoundError(f"Arquivo ausente na pasta portatil {pasta}: {arq}")
    df = pd.read_csv(pasta / "groups.csv", sep=";")
    if "cd_sessao" not in df.columns or "grupo" not in df.columns:
        raise ValueError(f"Contrato invalido em {pasta / 'groups.csv'}: colunas cd_sessao e grupo obrigatorias")
    params = ler_json(pasta / "params.json")
    return {
        "session_ids": df["cd_sessao"].tolist(),
        "cluster_labels": df["grupo"].to_numpy(dtype=int),
        **params,
    }


def _limpar_json_valor(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def portar_v1_para_portatil(
    root: Path | str,
    dest_data_dir: Path | str,
    dest_clusters_dir: Path | str,
) -> dict:
    """Converte os .pkl da v1.0 para formato portátil validando 100% de igualdade valor a valor."""
    raiz = Path(root)
    src_data = (raiz / "data" / "v1.0") if (raiz / "data" / "v1.0").exists() else (raiz / "data")
    src_clusters = (raiz / "clusters" / "v1.0") if (raiz / "clusters" / "v1.0").exists() else (raiz / "clusters")
    dst_data = Path(dest_data_dir)
    dst_clusters = Path(dest_clusters_dir)
    resumo = {"representacoes": {}, "agrupamentos": {}}

    for rep_nome, pkl_nome in REPS_PKL_V1.items():
        orig = carregar_representacao_pkl(src_data / pkl_nome)
        meta = {"vectorizer_type": orig.get("vectorizer_type", rep_nome)}
        if "feature_names" in orig:
            meta["feature_names"] = list(orig["feature_names"])
        pasta = salvar_representacao_portatil(dst_data / rep_nome, orig["X"], orig["session_ids"], meta=meta)
        port = carregar_representacao_portatil(pasta)
        assert np.array_equal(orig["X"], port["X"]), f"Divergência em X de {rep_nome}"
        assert list(orig["session_ids"]) == list(port["session_ids"]), f"Divergência em session_ids de {rep_nome}"
        if "feature_names" in orig:
            assert list(orig["feature_names"]) == list(port["feature_names"]), f"Divergência em feature_names de {rep_nome}"
        resumo["representacoes"][rep_nome] = str(pasta)

    for pkl_path in sorted(src_clusters.glob("*.pkl")):
        orig = carregar_agrupamento_pkl(pkl_path)
        params = {
            k: _limpar_json_valor(orig[k])
            for k in (
                "algorithm",
                "k",
                "n_clusters",
                "min_cluster_size",
                "min_samples",
                "metric",
                "noise_ratio",
                "silhouette_score",
                "inertia",
                "vectorizer_type",
            )
            if k in orig
        }
        pasta = salvar_agrupamento_portatil(
            dst_clusters / pkl_path.stem,
            orig["session_ids"],
            orig["cluster_labels"],
            params=params,
        )
        port = carregar_agrupamento_portatil(pasta)
        assert np.array_equal(
            np.asarray(orig["cluster_labels"], dtype=int),
            np.asarray(port["cluster_labels"], dtype=int),
        ), f"Divergência em cluster_labels de {pkl_path.name}"
        assert list(orig["session_ids"]) == list(port["session_ids"]), f"Divergência em session_ids de {pkl_path.name}"
        resumo["agrupamentos"][pkl_path.stem] = str(pasta)

    return resumo
