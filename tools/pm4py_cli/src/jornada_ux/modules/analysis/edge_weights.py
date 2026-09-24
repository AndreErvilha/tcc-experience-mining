"""
Módulo edge-weights: Distribuição de pesos das arestas do DFG com detecção de cotovelo,
para auxiliar na escolha do threshold do comando backbone.
"""

import os
import pm4py
import pandas as pd
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite


def _detect_elbow(contributions):
    """
    Detecta o cotovelo na curva de contribuições marginais.
    Retorna o índice (0-based) onde a maior queda absoluta ocorre.
    """
    if len(contributions) < 3:
        return None
    drops = [contributions[i] - contributions[i + 1] for i in range(len(contributions) - 1)]
    elbow_idx = drops.index(max(drops))
    return elbow_idx


def run_edge_weights(args):
    print(f"Lendo CSV de logs: {args.csv}")
    df_raw, df_filtered, df_event_log = prepare_event_log(args.csv, args.events)
    if df_event_log is None:
        return

    print("Descobrindo DFG e pesos das arestas...")
    try:
        dfg, start_activities, end_activities = pm4py.discover_dfg(df_event_log)
    except Exception as e:
        print(f"Erro ao descobrir DFG: {e}")
        return

    if not dfg:
        print("Nenhuma aresta encontrada no DFG.")
        return

    # Converter DFG (dict de (src, tgt) -> weight) para DataFrame
    rows = []
    for (src, tgt), weight in dfg.items():
        rows.append({"origem": src, "destino": tgt, "peso": weight})

    df = pd.DataFrame(rows)
    df = df.sort_values(by="peso", ascending=False).reset_index(drop=True)

    total_weight = df["peso"].sum()
    df["contribuicao_pct"] = (df["peso"] / total_weight) * 100
    df["acumulado_pct"] = df["contribuicao_pct"].cumsum()

    # Detectar cotovelo
    contributions = df["contribuicao_pct"].tolist()
    elbow_idx = _detect_elbow(contributions)

    # Imprimir tabela formatada
    top_n = min(len(df), args.top_n) if args.top_n else len(df)
    print(f"\n{'='*100}")
    print(f"  DISTRIBUICÃO DE PESOS DAS ARESTAS DO DFG")
    print(f"{'='*100}")
    print(f"  Total de arestas: {len(df)} | Peso total: {total_weight}")
    if elbow_idx is not None:
        print(f"  Cotovelo detectado na aresta #{elbow_idx + 1} "
              f"(maior queda de contribuicao: {contributions[elbow_idx]:.2f}% -> {contributions[elbow_idx + 1]:.2f}%)")
        print(f"  Sugestão de corte: --top_n {elbow_idx + 1} ou arestas com peso >= {df.iloc[elbow_idx]['peso']}")
    print(f"{'='*100}")
    print(f"  {'#':>4}  {'Origem':<45} {'Destino':<45} {'Peso':>6} {'Contrib%':>8} {'Acum%':>8}")
    print(f"  {'-'*4}  {'-'*45} {'-'*45} {'-'*6} {'-'*8} {'-'*8}")

    for i in range(top_n):
        row = df.iloc[i]
        marker = " <-- cotovelo" if elbow_idx is not None and i == elbow_idx else ""
        src_short = row["origem"][:43] + ".." if len(row["origem"]) > 45 else row["origem"]
        tgt_short = row["destino"][:43] + ".." if len(row["destino"]) > 45 else row["destino"]
        print(f"  {i+1:>4}  {src_short:<45} {tgt_short:<45} {row['peso']:>6} {row['contribuicao_pct']:>7.2f}% {row['acumulado_pct']:>7.1f}%{marker}")

    print(f"  {'='*100}")

    # Exportar CSV se solicitado
    if args.output_csv:
        output_csv = args.output_csv
    else:
        output_csv = resolve_output_file_path(args, args.csv, "analysis/edge-weights", "edge-weights", "csv")

    if output_csv and check_overwrite(output_csv, args):
        df.to_csv(output_csv, index=False)
        print(f"\nSucesso! Estatísticas salvas em CSV: {os.path.abspath(output_csv)}")

    # Gerar DFG visual se solicitado
    if args.dfg:
        png_path = resolve_output_file_path(args, args.csv, "analysis/edge-weights", "edge-weights-dfg", "png")
        dfg_path = resolve_output_file_path(args, args.csv, "analysis/edge-weights", "edge-weights-dfg", "dfg")
        if check_overwrite([png_path, dfg_path], args):
            try:
                pm4py.save_vis_dfg(dfg, start_activities, end_activities, png_path, rankdir='TB')
                pm4py.write_dfg(dfg, start_activities, end_activities, dfg_path)
                print(f"DFG visual salvo em: {os.path.abspath(png_path)}")
                print(f"DFG serializado em: {os.path.abspath(dfg_path)}")
            except Exception as e:
                print(f"Erro ao gerar DFG visual: {e}")
