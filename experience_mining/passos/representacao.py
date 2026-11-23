"""Passos de vetorização (TF-IDF, MiniLM, Gemini) com registro por nome."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from experience_mining.configuracao import ConfigRepresentacoes, REP_JSON_PARA_MODELO
from experience_mining.io import salvar_representacao_pkl, salvar_representacao_portatil


def _persistir_representacao(
    output_path: Path | str | None,
    data_dict: dict,
    portatil: bool = False,
) -> None:
    if output_path is None:
        return
    out = Path(output_path)
    if portatil:
        meta = {"vectorizer_type": data_dict["vectorizer_type"]}
        if "feature_names" in data_dict:
            meta["feature_names"] = data_dict["feature_names"]
        salvar_representacao_portatil(out, data_dict["X"], data_dict["session_ids"], meta=meta)
        print(f"[+] Representacao portatil salva em: {out}")
        return

    salvar_representacao_pkl(out, data_dict)
    np_path = str(out).replace(".pkl", ".npy") if str(out).endswith(".pkl") else f"{out}.npy"
    np.save(
        np_path,
        {
            "X": data_dict["X"],
            "labels": data_dict.get("primary_labels"),
            "experiences": data_dict.get("experiences"),
            "session_ids": data_dict["session_ids"],
        },
    )
    print(f"[+] Vetores salvos em: {out} e {np_path}")


def vetorizar_tfidf(
    sessions_df: pd.DataFrame,
    output_path: Path | str | None = None,
    max_features: int | None = None,
    min_df: int = 1,
    portatil: bool = False,
    **_,
) -> dict:
    texts = sessions_df["text_representation"].fillna("").astype(str).tolist()
    session_ids = sessions_df["cd_sessao"].tolist()
    primary_labels = sessions_df["primary_label"].tolist() if "primary_label" in sessions_df.columns else None
    experiences = sessions_df["experience"].tolist() if "experience" in sessions_df.columns else None

    print('[*] Vetorizando com TF-IDF (token_pattern=r"\\S+", lowercase=False)...')
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"\S+",
        max_features=max_features,
        min_df=min_df,
    )
    X_sparse = vectorizer.fit_transform(texts)
    X = X_sparse.toarray().astype(np.float32)
    feature_names = vectorizer.get_feature_names_out().tolist()
    print(f"[+] Matriz TF-IDF construida com shape: {X.shape}")

    data_dict = {
        "X": X,
        "feature_names": feature_names,
        "session_ids": session_ids,
        "primary_labels": primary_labels,
        "experiences": experiences,
        "vectorizer_type": "TF-IDF",
        "df": sessions_df,
    }
    _persistir_representacao(output_path, data_dict, portatil=portatil)
    return data_dict


def _gerar_embeddings_minilm(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    print(f"[*] Carregando modelo local: {model_name}...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    print(f"[*] Gerando embeddings para {len(texts)} sessoes...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def _gerar_embeddings_gemini(
    texts: list[str],
    api_key: str | None = None,
    model_name: str = "models/gemini-embedding-001",
    batch_size: int = 50,
) -> np.ndarray:
    print(f"[*] Configurando Gemini API ({model_name})...")
    import google.generativeai as genai

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        from dotenv import load_dotenv

        load_dotenv(override=False)
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY ou GOOGLE_API_KEY nao encontrada nas variaveis de ambiente.")
    genai.configure(api_key=key)

    all_embeddings = []
    print(f"[*] Solicitando embeddings para {len(texts)} sessoes via Gemini...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Gemini Batches"):
        batch = [t if t.strip() else "empty_session" for t in texts[i : i + batch_size]]
        response = genai.embed_content(model=model_name, content=batch, task_type="CLUSTERING")
        if hasattr(response, "embeddings"):
            batch_emb = response.embeddings
        elif isinstance(response, dict) and "embedding" in response:
            batch_emb = response["embedding"]
        else:
            batch_emb = response
        if isinstance(batch_emb[0], dict) and "values" in batch_emb[0]:
            batch_vecs = [item["values"] for item in batch_emb]
        else:
            batch_vecs = batch_emb
        all_embeddings.extend(batch_vecs)

    X = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _montar_embeddings(sessions_df: pd.DataFrame, X: np.ndarray, vec_name: str) -> dict:
    session_ids = sessions_df["cd_sessao"].tolist()
    primary_labels = sessions_df["primary_label"].tolist() if "primary_label" in sessions_df.columns else None
    experiences = sessions_df["experience"].tolist() if "experience" in sessions_df.columns else None
    return {
        "X": X,
        "session_ids": session_ids,
        "primary_labels": primary_labels,
        "experiences": experiences,
        "vectorizer_type": vec_name,
        "df": sessions_df,
    }


def vetorizar_minilm(
    sessions_df: pd.DataFrame,
    output_path: Path | str | None = None,
    minilm_model_name: str = "all-MiniLM-L6-v2",
    model_name: str | None = None,
    batch_size: int = 32,
    device: str = "cpu",
    portatil: bool = False,
    **_,
) -> dict:
    nome_modelo = model_name or minilm_model_name
    texts = sessions_df["text_representation"].fillna("empty").astype(str).tolist()
    X = _gerar_embeddings_minilm(texts, model_name=nome_modelo, batch_size=batch_size, device=device)
    print(f"[+] Embeddings gerados com shape: {X.shape}")
    data_dict = _montar_embeddings(sessions_df, X, f"Embedding-{nome_modelo}")
    _persistir_representacao(output_path, data_dict, portatil=portatil)
    return data_dict


def vetorizar_gemini(
    sessions_df: pd.DataFrame,
    output_path: Path | str | None = None,
    gemini_model_name: str = "models/gemini-embedding-001",
    model_name: str | None = None,
    batch_size: int = 32,
    api_key: str | None = None,
    portatil: bool = False,
    **_,
) -> dict:
    nome_modelo = model_name or gemini_model_name
    texts = sessions_df["text_representation"].fillna("empty").astype(str).tolist()
    X = _gerar_embeddings_gemini(texts, api_key=api_key, model_name=nome_modelo, batch_size=batch_size)
    print(f"[+] Embeddings gerados com shape: {X.shape}")
    data_dict = _montar_embeddings(sessions_df, X, "Embedding-Gemini")
    _persistir_representacao(output_path, data_dict, portatil=portatil)
    return data_dict


REPRESENTACOES = {
    "tfidf": vetorizar_tfidf,
    "minilm": vetorizar_minilm,
    "gemini": vetorizar_gemini,
}


def executar_representacoes(secao: ConfigRepresentacoes, caminho_sessoes: Path) -> dict[str, Path]:
    """Executa a etapa de representações conforme a seção de configuração."""
    if secao.itens:
        caminhos = {REP_JSON_PARA_MODELO[k]: item.path for k, item in secao.itens.items()}
        gerar_itens = [item for item in secao.itens.values() if item.modo == "gerar"]
        if not gerar_itens:
            return caminhos
        sessions_df = pd.read_csv(caminho_sessoes)
        for item in gerar_itens:
            REPRESENTACOES[item.nome](
                sessions_df,
                output_path=item.path,
                portatil=True,
                **item.params,
            )
        return caminhos

    caminhos = {
        "tf_idf": secao.tfidf_path,
        "minilm": secao.minilm_path,
        "gemini": secao.gemini_path,
    }
    if not secao.gerar:
        return caminhos

    sessions_df = pd.read_csv(caminho_sessoes)
    REPRESENTACOES["tfidf"](
        sessions_df,
        output_path=secao.tfidf_path,
        max_features=secao.max_features,
        min_df=secao.min_df,
    )
    REPRESENTACOES["minilm"](
        sessions_df,
        output_path=secao.minilm_path,
        minilm_model_name=secao.minilm_model_name,
        batch_size=secao.batch_size,
    )
    if not secao.gemini_path.exists():
        print(f"[!] {secao.gemini_path} nao encontrado. Pulei a etapa do Gemini.")
        print("    Defina GEMINI_API_KEY e gere previamente caso queira regerar os embeddings Gemini.")
    return caminhos
