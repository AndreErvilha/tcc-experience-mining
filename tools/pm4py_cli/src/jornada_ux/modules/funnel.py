import os
import pandas as pd
import numpy as np
import pm4py
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite

def find_most_probable_path(df_event_log, max_steps=12):
    """
    Encontra o caminho estocasticamente mais provável (Most Probable Path)
    navegando pelas transições de maior probabilidade relativa P(B|A) no DFG.
    """
    try:
        dfg, start_activities, end_activities = pm4py.discover_dfg(df_event_log)
    except Exception:
        return []

    if not dfg or not start_activities:
        return []

    # Atividade inicial de maior probabilidade/frequência
    curr_act = max(start_activities, key=start_activities.get)
    path = [curr_act]
    visited = {curr_act}

    while len(path) < max_steps:
        # Encontrar todas as transições de saída a partir de curr_act
        out_transitions = [(tgt, count) for (src, tgt), count in dfg.items() if src == curr_act]
        if not out_transitions:
            break

        # Ordenar por maior frequência (maior probabilidade de transição)
        out_transitions.sort(key=lambda x: x[1], reverse=True)

        # Escolher a próxima atividade mais provável não visitada (para evitar loops)
        next_act = None
        for tgt, _ in out_transitions:
            if tgt not in visited:
                next_act = tgt
                break

        if not next_act:
            break

        path.append(next_act)
        visited.add(next_act)
        curr_act = next_act

        if curr_act in end_activities:
            break

    return path

def discover_heuristic_funnel_steps(df_event_log, top_n=8, use_most_probable=True):
    """
    Descobre euristicamente a sequência principal do funil.
    Por padrão, utiliza o Caminho de Maior Probabilidade (Most Probable Path).
    """
    if use_most_probable:
        path = find_most_probable_path(df_event_log, max_steps=top_n)
        if path:
            return path

    # Fallback para a variante de maior frequência
    variants_dict = pm4py.get_variants(df_event_log)
    if not variants_dict:
        return []
    
    sorted_variants = sorted(variants_dict.items(), key=lambda x: x[1], reverse=True)
    top_variant = sorted_variants[0][0]
    
    steps = []
    for act in top_variant:
        if not steps or steps[-1] != act:
            steps.append(act)
            
    return steps[:top_n] if len(steps) > top_n else steps


def calculate_funnel_metrics(df_event_log, steps=None):
    """
    Calcula as métricas do funil de conversão para uma lista de passos (telas/eventos).
    """
    if not steps:
        steps = discover_heuristic_funnel_steps(df_event_log)
        
    if not steps:
        print("Erro: Não foi possível determinar os passos do funil.")
        return None, []

    # Mapear sessões e seus eventos ordenados por timestamp
    cases = df_event_log.groupby('case:concept:name')
    
    stage_case_ids = {i: set() for i in range(len(steps))}
    stage_timestamps = {i: {} for i in range(len(steps))}

    for case_id, group in cases:
        group_sorted = group.sort_values(by='time:timestamp')
        events = group_sorted['concept:name'].tolist()
        timestamps = group_sorted['time:timestamp'].tolist()
        
        current_step_idx = 0
        last_found_time = None
        
        for act, ts in zip(events, timestamps):
            target_step = steps[current_step_idx]
            
            if target_step == act:
                stage_case_ids[current_step_idx].add(case_id)
                stage_timestamps[current_step_idx][case_id] = ts
                current_step_idx += 1
                if current_step_idx >= len(steps):
                    break

    # Construir tabela do funil
    funnel_rows = []
    initial_cases = len(stage_case_ids[0]) if len(steps) > 0 else 0
    
    for i, step in enumerate(steps):
        current_cases = len(stage_case_ids[i])
        prev_cases = len(stage_case_ids[i-1]) if i > 0 else current_cases
        
        overall_conversion = (current_cases / initial_cases * 100) if initial_cases > 0 else 0
        step_conversion = (current_cases / prev_cases * 100) if prev_cases > 0 else 0
        drop_off = prev_cases - current_cases if i > 0 else 0
        drop_off_pct = (drop_off / prev_cases * 100) if prev_cases > 0 else 0
        
        # Calcular tempo de transição a partir da etapa anterior
        durations = []
        if i > 0:
            for c_id in stage_case_ids[i]:
                if c_id in stage_timestamps[i-1]:
                    t_prev = stage_timestamps[i-1][c_id]
                    t_curr = stage_timestamps[i][c_id]
                    dur = (t_curr - t_prev).total_seconds()
                    if dur >= 0:
                        durations.append(dur)
                        
        mean_time = np.mean(durations) if durations else 0.0
        median_time = np.median(durations) if durations else 0.0
        
        funnel_rows.append({
            "etapa_idx": i + 1,
            "etapa_nome": step,
            "casos_atingiram": current_cases,
            "conversao_acumulada_pct": round(overall_conversion, 2),
            "conversao_etapa_pct": round(step_conversion, 2),
            "desistencias_dropoff": drop_off,
            "taxa_evasao_pct": round(drop_off_pct, 2),
            "tempo_medio_etapa_s": round(mean_time, 2),
            "tempo_mediano_etapa_s": round(median_time, 2)
        })
        
    df_funnel = pd.DataFrame(funnel_rows)
    return df_funnel, steps

def print_ascii_funnel(df_funnel):
    """Exibe um gráfico de funil em formato de texto no terminal."""
    print("\n" + "="*80)
    print("                      FUNIL DE CONVERSAO DE JORNADA UX                      ")
    print("="*80)
    
    max_cases = df_funnel['casos_atingiram'].max() if len(df_funnel) > 0 else 1
    bar_max_len = 35
    
    for idx, row in df_funnel.iterrows():
        cases = row['casos_atingiram']
        bar_len = int((cases / max_cases) * bar_max_len) if max_cases > 0 else 0
        bar_str = "=" * bar_len
        
        print(f"\nEtapa #{row['etapa_idx']}: {row['etapa_nome']}")
        print(f"   Progresso : [{bar_str:<35}] {cases} sessoes ({row['conversao_acumulada_pct']}%)")
        if idx > 0:
            print(f"   Retencao  : {row['conversao_etapa_pct']}% da etapa anterior | Drop-off: {row['desistencias_dropoff']} ({row['taxa_evasao_pct']}%)")
            print(f"   Tempo Etapa: Medio {row['tempo_medio_etapa_s']}s | Mediano {row['tempo_mediano_etapa_s']}s")
    print("\n" + "="*80)


def generate_funnel_chart_png(df_funnel, output_png_path):
    """Gera um gráfico visual PNG do funil usando matplotlib se disponível."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        stages = [f"#{r['etapa_idx']} {r['etapa_nome'][:30]}" for _, r in df_funnel.iterrows()]
        values = df_funnel['casos_atingiram'].tolist()
        percentages = df_funnel['conversao_acumulada_pct'].tolist()
        
        # Inverter para desenhar de cima para baixo
        stages = stages[::-1]
        values = values[::-1]
        percentages = percentages[::-1]
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(stages)))
        bars = ax.barh(stages, values, color=colors, edgecolor='navy')
        
        for bar, val, pct in zip(bars, values, percentages):
            ax.text(bar.get_width() + (max(values)*0.02), bar.get_y() + bar.get_height()/2,
                    f"{val} sessões ({pct}%)", va='center', ha='left', fontweight='bold', color='#1a2a3a')
                    
        ax.set_title("Funil de Conversão de Jornada UX", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Quantidade de Sessões / Usuários", fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_png_path, dpi=300)
        plt.close()
        return True
    except Exception as e:
        print(f"Aviso: Não foi possível gerar imagem PNG do gráfico ({e}). Salvação no formato CSV mantida.")
        return False

def run_funnel(args):
    output_csv = resolve_output_file_path(args, args.csv, "funnel", "funil", "csv", explicit_override=args.output_csv)
    output_png = resolve_output_file_path(args, args.csv, "funnel", "funil", "png", explicit_override=getattr(args, 'output_png', None))
    
    if not check_overwrite([output_csv, output_png], args):
        return

    print(f"Lendo e formatando CSV de logs: {args.csv}")
    _, _, df_event_log = prepare_event_log(args.csv, args.events)
    if df_event_log is None:
        return

    steps = getattr(args, 'steps', None)
    if steps:
        print(f"Calculando funil de conversão para os passos especificados: {steps}")
    else:
        print("Buscando euristicamente o caminho/passos do funil principal...")

    df_funnel, used_steps = calculate_funnel_metrics(df_event_log, steps=steps)
    if df_funnel is None:
        return

    print_ascii_funnel(df_funnel)

    # Salvar CSV
    df_funnel.to_csv(output_csv, index=False)
    print(f"\nSucesso! Tabela detalhada do funil salva em CSV: {os.path.abspath(output_csv)}")

    # Gerar gráfico PNG
    if generate_funnel_chart_png(df_funnel, output_png):
        print(f"Sucesso! Gráfico de funil salvo em PNG: {os.path.abspath(output_png)}")
