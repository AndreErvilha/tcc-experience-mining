"""
Módulo de Análise de Loops por Atividade.

Identifica quais atividades específicas geram loops (repetições dentro
de uma mesma sessão) e as ranqueia por frequência, revelando os pontos
de fricção mais comuns da jornada.

Gera também o DFG visual (PNG + .dfg) das transições que compõem os loops.
"""
import os
import pandas as pd
import pm4py
from collections import defaultdict
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite


def detect_loops(events, min_repetitions=2):
    """
    Detecta loops em uma sequência de eventos.

    Um "loop" ocorre quando a mesma atividade aparece ao menos
    min_repetitions vezes dentro da sessão.

    Retorna um dicionário {atividade: total_ocorrencias}.
    """
    from collections import Counter
    counts = Counter(events)
    return {act: cnt for act, cnt in counts.items() if cnt >= min_repetitions}


def extract_loop_edges(events, loop_activities):
    """
    Extrai as arestas que compõem a "zona de loop" de uma sessão.

    A zona de loop vai do primeiro evento que é uma atividade loopante
    até o último evento que é uma atividade loopante (inclusive).

    Retorna uma lista de pares (src, tgt) consecutivos nessa zona.
    """
    # Encontrar os índices do primeiro e último evento loopante
    loop_indices = [
        i for i, evt in enumerate(events)
        if evt in loop_activities
    ]
    if not loop_indices:
        return []

    start_idx = loop_indices[0]
    end_idx = loop_indices[-1]

    edges = []
    for i in range(start_idx, end_idx):
        edges.append((events[i], events[i + 1]))

    return edges


def build_dfg_from_edges(edge_counter):
    """
    Constrói um DFG a partir de um contador de arestas.

    edge_counter: {(src, tgt): count}

    Retorna (dfg_dict, start_activities, end_activities).
    """
    dfg = dict(edge_counter)

    sources = set()
    targets = set()
    for src, tgt in dfg:
        sources.add(src)
        targets.add(tgt)

    start_activities = {a: 1 for a in (sources - targets)}
    end_activities = {a: 1 for a in (targets - sources)}

    # Se o grafo for um ciclo puro (sem start/end distintos), usa todos os nós
    if not start_activities and not end_activities and sources:
        first = next(iter(sources))
        start_activities = {first: 1}
        end_activities = {first: 1}

    return dfg, start_activities, end_activities


def run_loops(args):
    """Ponto de entrada do comando analysis loops."""
    output_csv = resolve_output_file_path(
        args, args.csv, "analysis/loops", "loops", "csv",
        explicit_override=getattr(args, 'output_csv', None)
    )

    df_raw, df_filtered, df_event_log = prepare_event_log(args.csv, args.events)
    if df_event_log is None:
        return

    min_repetitions = getattr(args, 'min_repetitions', 2)
    top_k = getattr(args, 'top_k', 20)

    print(f"\nTotal de registros lidos: {len(df_raw)}")
    print(f"Registros após filtragem de eventos ({args.events}): {len(df_filtered)}")
    print(f"\n=== Detectando loops (atividades com >= {min_repetitions} ocorrências na mesma sessão) ===")

    sessions_with_loop = defaultdict(int)
    total_revisits = defaultdict(int)
    max_rep = defaultdict(int)
    total_occurrences = defaultdict(int)
    detail_rows = []

    # Contador de arestas para o DFG de loops
    from collections import Counter as EdgeCounter
    loop_edge_counter = EdgeCounter()

    total_sessions = 0
    sessions_with_any_loop = 0

    # Cache: eventos por sessão para extrair arestas de loop
    session_events = {}

    for case_id, group in df_event_log.groupby('case:concept:name'):
        total_sessions += 1
        events = group.sort_values(by='time:timestamp')['concept:name'].tolist()
        session_events[case_id] = events

        loops = detect_loops(events, min_repetitions)
        if loops:
            sessions_with_any_loop += 1
            loop_activities = set(loops.keys())

            # Extrair arestas da zona de loop
            loop_edges = extract_loop_edges(events, loop_activities)
            for edge in loop_edges:
                loop_edge_counter[edge] += 1

            for activity, count in loops.items():
                sessions_with_loop[activity] += 1
                revisits = count - 1
                total_revisits[activity] += revisits
                total_occurrences[activity] += count
                if count > max_rep[activity]:
                    max_rep[activity] = count

                detail_rows.append({
                    "cd_sessao": case_id,
                    "atividade": activity,
                    "repeticoes_na_sessao": count,
                    "revisitas": revisits,
                })

    if not sessions_with_loop:
        print("Nenhum loop detectado com os parâmetros atuais.")
        return

    # Construir DataFrame principal
    rows = []
    for activity in sessions_with_loop:
        sessoes = sessions_with_loop[activity]
        revisitas = total_revisits[activity]
        rows.append({
            "atividade": activity,
            "sessoes_com_loop": sessoes,
            "pct_sessoes": round(sessoes / total_sessions * 100, 1),
            "total_revisitas": revisitas,
            "media_revisitas_por_sessao_afetada": round(revisitas / sessoes, 2),
            "max_repeticoes_em_uma_sessao": max_rep[activity],
            "total_ocorrencias_absolutas": total_occurrences[activity],
        })

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(by="sessoes_com_loop", ascending=False).reset_index(drop=True)

    # Exibir
    df_display = df_out.head(top_k)
    print(f"\nSessões analisadas: {total_sessions}")
    print(f"Sessões com ao menos 1 loop: {sessions_with_any_loop} ({round(sessions_with_any_loop/total_sessions*100,1)}%)")
    print(f"Atividades distintas com loop: {len(df_out)}")
    print(f"\n--- TOP {min(top_k, len(df_display))} ATIVIDADES COM MAIS LOOPS ---")
    for _, row in df_display.iterrows():
        print(f"  [{row['sessoes_com_loop']:>3d} sessões | {row['total_revisitas']:>3d} revisitas | máx {row['max_repeticoes_em_uma_sessao']}x]  {row['atividade']}")

    # --- DFG visualization ---
    print(f"\n=== Gerando DFG das transições de loop ===")
    dfg_loops, start_loops, end_loops = build_dfg_from_edges(loop_edge_counter)
    print(f"DFG de loops gerado: {len(dfg_loops)} arestas, peso total = {sum(dfg_loops.values())}")

    # Output paths
    png_dfg = resolve_output_file_path(args, args.csv, "analysis/loops", "loops-dfg", "png")
    dfg_file = resolve_output_file_path(args, args.csv, "analysis/loops", "loops-dfg", "dfg")
    detail_csv = resolve_output_file_path(args, args.csv, "analysis/loops", "loops-detalhe", "csv")

    files_to_check = [output_csv, png_dfg, dfg_file]
    if not check_overwrite(files_to_check, args):
        return

    # Salvar CSV principal
    df_out.to_csv(output_csv, index=False)
    print(f"CSV salvo em: {os.path.abspath(output_csv)}")

    # Salvar DFG PNG
    try:
        pm4py.save_vis_dfg(dfg_loops, start_loops, end_loops, png_dfg, rankdir='TB')
        print(f"DFG PNG salvo em: {os.path.abspath(png_dfg)}")
    except Exception as e:
        print(f"Erro ao gerar PNG do DFG: {e}")

    # Salvar DFG .dfg
    try:
        pm4py.write_dfg(dfg_loops, start_loops, end_loops, dfg_file)
        print(f"DFG (.dfg) salvo em: {os.path.abspath(dfg_file)}")
    except Exception as e:
        print(f"Erro ao exportar .dfg: {e}")

    # Salvar detalhamento
    if detail_rows:
        df_detail = pd.DataFrame(detail_rows)
        df_detail = df_detail.sort_values(by=["atividade", "cd_sessao"]).reset_index(drop=True)
        df_detail.to_csv(detail_csv, index=False)
        print(f"Detalhamento por sessão salvo em: {os.path.abspath(detail_csv)}")
