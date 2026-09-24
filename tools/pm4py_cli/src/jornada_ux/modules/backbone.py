"""
Módulo Backbone (Espinha Dorsal do Processo).

Extrai uma Rede de Petri simplificada a partir apenas das transições
de maior peso (frequência) do DFG, revelando a "espinha dorsal"
do processo — os caminhos mais relevantes da jornada.
"""
import os
import pm4py
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite


def filter_dfg_by_weight(dfg, start_activities, end_activities,
                         top_pct=None, min_weight=None, top_n=None):
    """
    Filtra o DFG mantendo apenas as arestas mais relevantes por peso.

    Parâmetros (mutuamente exclusivos, prioridade na ordem abaixo):
        top_pct:   float (0-100). Mantém arestas até atingir X% do peso total,
                   ordenadas descendentemente.
        min_weight: int. Mantém arestas com peso >= min_weight.
        top_n:     int. Mantém as top-N arestas de maior peso.

    Retorna:
        (dfg_filtered, start_filtered, end_filtered, kept_count, total_count, kept_weight, total_weight)
    """
    if not dfg:
        return {}, set(), set(), 0, 0, 0, 0

    total_weight = sum(dfg.values())
    total_count = len(dfg)

    # Ordenar arestas por peso decrescente
    sorted_edges = sorted(dfg.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None:
        kept = {edge: weight for edge, weight in sorted_edges[:top_n]}
    elif min_weight is not None:
        kept = {edge: weight for edge, weight in sorted_edges if weight >= min_weight}
    elif top_pct is not None:
        threshold = total_weight * (top_pct / 100.0)
        cumulative = 0
        kept = {}
        for edge, weight in sorted_edges:
            kept[edge] = weight
            cumulative += weight
            if cumulative >= threshold:
                break
    else:
        # Sem filtro: mantém tudo
        kept = dict(dfg)

    # Ajustar start_activities e end_activities para conter apenas
    # atividades que aparecem no DFG filtrado
    activities_in_dfg = set()
    for src, tgt in kept:
        activities_in_dfg.add(src)
        activities_in_dfg.add(tgt)

    start_filtered = {a: c for a, c in start_activities.items() if a in activities_in_dfg}
    end_filtered = {a: c for a, c in end_activities.items() if a in activities_in_dfg}

    kept_weight = sum(kept.values())
    kept_count = len(kept)

    return kept, start_filtered, end_filtered, kept_count, total_count, kept_weight, total_weight


def run_backbone(args):
    """Ponto de entrada do comando backbone."""
    df_raw, df_filtered, df_event_log = prepare_event_log(args.csv, args.events)
    if df_event_log is None:
        return

    # Determinar o critério de filtro
    filter_mode = None
    if getattr(args, 'top_pct', None) is not None:
        filter_mode = 'top_pct'
    if getattr(args, 'min_weight', None) is not None:
        filter_mode = 'min_weight'
    if getattr(args, 'top_n', None) is not None:
        filter_mode = 'top_n'

    print(f"\nTotal de registros lidos: {len(df_raw)}")
    print(f"Registros após filtragem de eventos ({args.events}): {len(df_filtered)}")

    # 1. Descobrir DFG completo com pesos
    print("\n=== Descobrindo DFG (Directly-Follows Graph) com pesos ===")
    try:
        dfg, start_activities, end_activities = pm4py.discover_dfg(df_event_log)
        print(f"DFG descoberto: {len(dfg)} arestas, peso total = {sum(dfg.values())}")
    except Exception as e:
        print(f"Erro ao descobrir DFG: {e}")
        return

    # 2. Filtrar por peso
    print(f"\n=== Filtrando arestas do DFG por peso ===")
    if filter_mode == 'top_pct':
        print(f"Critério: top_pct = {args.top_pct}% do peso total acumulado")
    elif filter_mode == 'min_weight':
        print(f"Critério: min_weight = {args.min_weight} ocorrências")
    elif filter_mode == 'top_n':
        print(f"Critério: top_n = {args.top_n} arestas mais frequentes")
    else:
        print("Critério: nenhum (mantendo todas as arestas)")

    dfg_filt, start_filt, end_filt, kept_count, total_count, kept_weight, total_weight = \
        filter_dfg_by_weight(
            dfg, start_activities, end_activities,
            top_pct=args.top_pct if filter_mode == 'top_pct' else None,
            min_weight=args.min_weight if filter_mode == 'min_weight' else None,
            top_n=args.top_n if filter_mode == 'top_n' else None,
        )

    pct_edges = (kept_count / total_count * 100) if total_count else 0
    pct_weight = (kept_weight / total_weight * 100) if total_weight else 0
    print(f"Arestas mantidas: {kept_count}/{total_count} ({pct_edges:.1f}%)")
    print(f"Peso mantido:    {kept_weight}/{total_weight} ({pct_weight:.1f}%)")

    if kept_count == 0:
        print("Erro: Nenhuma aresta restante após filtragem. Ajuste os parâmetros de corte.")
        return

    # 3. Filtrar event log para manter apenas atividades do DFG filtrado
    activities_in_backbone = set()
    for src, tgt in dfg_filt:
        activities_in_backbone.add(src)
        activities_in_backbone.add(tgt)
    df_event_log_filtered = df_event_log[df_event_log['concept:name'].isin(activities_in_backbone)]

    # 4. Converter DFG filtrado para Rede de Petri (via algoritmo escolhido)
    algo = getattr(args, 'algo', 'dfg_convert')
    print(f"\n=== Gerando Rede de Petri (backbone) via algoritmo: {algo} ===")
    try:
        if algo == 'dfg_convert':
            net, initial_marking, final_marking = pm4py.convert_to_petri_net(
                dfg_filt, start_filt, end_filt
            )
        elif algo == 'inductive':
            net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(df_event_log_filtered)
        elif algo == 'alpha':
            net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(df_event_log_filtered)
        elif algo == 'heuristics':
            net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(df_event_log_filtered)
        else:
            print(f"Erro: algoritmo '{algo}' desconhecido.")
            return
        print("Rede de Petri (backbone) gerada com sucesso!")
    except Exception as e:
        print(f"Erro ao gerar Rede de Petri: {e}")
        return

    # 4. Exportar
    png_output = resolve_output_file_path(args, args.csv, "backbone", "backbone", "png")
    pnml_output = resolve_output_file_path(args, args.csv, "backbone", "backbone", "pnml")

    if not check_overwrite([png_output, pnml_output], args):
        return

    # Exportar PNML
    try:
        print(f"Salvando PNML em: {os.path.abspath(pnml_output)}")
        pm4py.write_pnml(net, initial_marking, final_marking, pnml_output)
        print("PNML exportado com sucesso!")
    except Exception as e:
        print(f"Erro ao exportar PNML: {e}")

    # Exportar PNG
    try:
        print(f"Salvando PNG em: {os.path.abspath(png_output)}")
        pm4py.save_vis_petri_net(net, initial_marking, final_marking, png_output, rankdir='TB')
        print("PNG exportado com sucesso!")
    except Exception as e:
        print(f"Erro ao exportar PNG: {e}")

    # Resumo
    print(f"\n=== Resumo ===")
    print(f"Lugares (places):      {len(net.places)}")
    print(f"Transições:            {len(net.transitions)}")
    print(f"Arcos:                 {len(net.arcs)}")
    print(f"Arestas DFG mantidas:  {kept_count} de {total_count}")
    print(f"Peso preservado:       {pct_weight:.1f}%")
    print(f"Saída PNML:            {os.path.abspath(pnml_output)}")
    print(f"Saída PNG:             {os.path.abspath(png_output)}")
