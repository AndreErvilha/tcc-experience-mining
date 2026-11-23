"""Tokenização de eventos e derivação de rótulo principal e experiência."""
from pathlib import Path
import os

import pandas as pd

from experience_mining.nucleo.jornadas import ABANDONO_COMPOSTAS, COMPOSTAS


def format_event_token(row, include_product: bool = True) -> str:
    event_type = str(row.get("cd_evento_tipo", "")).strip()
    page = str(row.get("page_name", "")).strip()
    btn = str(row.get("button_text", "")).strip()
    prod = str(row.get("cd_produto", "")).strip()

    if btn == "nan" or not btn or btn.lower() == "none":
        btn_str = ""
    else:
        btn_str = f"_{btn}"

    if event_type == "page_view":
        token = f"view_{page}"
    elif event_type == "click_button":
        token = f"click_{page}{btn_str}"
    else:
        token = f"{event_type}_{page}{btn_str}"

    if include_product and prod and prod != "nan":
        return f"{prod}:{token}"
    return token


def extract_primary_label(
    label_series: pd.Series,
    compostas: dict[str, list[str]] = COMPOSTAS,
    abandono_compostas: dict[str, list[str]] = ABANDONO_COMPOSTAS,
) -> str:
    """Deriva o rótulo principal da sessão a partir das flags is_* (Tabela 1 revisada)."""
    is_cols = [c for c in label_series.index if c.startswith("is_")]
    active = [c.replace("is_", "") for c in is_cols if label_series[c] == 1]
    success = [a for a in active if "sem_converter" not in a]
    for comp, partes in compostas.items():
        if comp in success:
            success = [a for a in success if a not in partes]
    if success:
        return "+".join(success)
    for comp, partes in abandono_compostas.items():
        if comp in active:
            active = [a for a in active if a not in partes]
    if active:
        return "+".join(active)
    return "neutro"


def mapas_absorcao_de_regras(
    labeling_params: list[dict],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Constrói mapas de absorção (compostas de sucesso e de abandono) a partir de labeling_params."""
    compostas: dict[str, list[str]] = {}
    abandono_compostas: dict[str, list[str]] = {}
    for regra in labeling_params:
        absorvidas = regra.get("absorbs_label_cols") or []
        if not absorvidas:
            continue
        chave = regra["label_col"].removeprefix("is_")
        partes = [col.removeprefix("is_") for col in absorvidas]
        if "sem_converter" in chave:
            abandono_compostas[chave] = partes
        else:
            compostas[chave] = partes
    return compostas, abandono_compostas


def classificar_experiencia(label_summary: dict[str, int]) -> str:
    has_success = any(v == 1 for k, v in label_summary.items() if "sem_converter" not in k)
    if has_success:
        return "success"
    has_attrition = any(v == 1 for k, v in label_summary.items() if "sem_converter" in k)
    return "attrition" if has_attrition else "none"


def processar_eventos_para_sessoes(
    df: pd.DataFrame,
    include_product: bool = True,
    compostas: dict[str, list[str]] = COMPOSTAS,
    abandono_compostas: dict[str, list[str]] = ABANDONO_COMPOSTAS,
) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(by=["cd_sessao", "timestamp"])

    df["event_token"] = df.apply(lambda r: format_event_token(r, include_product), axis=1)
    is_cols = [c for c in df.columns if c.startswith("is_")]

    grouped_rows = []
    for sess_id, group in df.groupby("cd_sessao", sort=False):
        tokens = group["event_token"].tolist()
        text_repr = " ".join(tokens)
        label_summary = {col: int(group[col].max()) if len(group[col]) > 0 else 0 for col in is_cols}
        primary_label = extract_primary_label(
            pd.Series(label_summary),
            compostas=compostas,
            abandono_compostas=abandono_compostas,
        )
        grouped_rows.append(
            {
                "cd_sessao": sess_id,
                "text_representation": text_repr,
                "num_events": len(tokens),
                "primary_label": primary_label,
                "experience": classificar_experiencia(label_summary),
                **label_summary,
            }
        )

    return pd.DataFrame(grouped_rows)


def processar_eventos_csv(
    csv_path: Path | str,
    output_csv: Path | str | None = None,
    include_product: bool = True,
) -> pd.DataFrame:
    print(f"[*] Carregando dataset de eventos: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[*] Formatando tokens para {len(df)} eventos...")
    print("[*] Agrupando por sessao (cd_sessao)...")
    result_df = processar_eventos_para_sessoes(df, include_product=include_product)
    print(f"[+] Sucesso! {len(result_df)} sessoes geradas.")
    if output_csv:
        out = Path(output_csv)
        os.makedirs(out.parent.resolve(), exist_ok=True)
        result_df.to_csv(out, index=False)
        print(f"[+] Salvo em: {output_csv}")
    return result_df
