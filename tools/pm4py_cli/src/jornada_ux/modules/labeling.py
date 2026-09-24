import os
import pandas as pd
import numpy as np
import pm4py
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite
from jornada_ux.modules.analysis.filter import apply_all_filters


def generate_heuristic_labels(df_event_log, target_event=None, bounce_max_events=2, slow_threshold_p85=None):
    """
    Classifica cada sessão/caso do dataset não rotulado atribuindo rótulos heurísticos de comportamento.
    """
    cases = df_event_log.groupby('case:concept:name')
    
    if not target_event:
        end_activities = pm4py.get_end_activities(df_event_log)
        if end_activities:
            target_event = max(end_activities, key=end_activities.get)
            print(f"Evento de conversão alvo não informado. Definido euristicamente como: '{target_event}'")

    case_durations = []
    for _, group in cases:
        dur = (group['time:timestamp'].max() - group['time:timestamp'].min()).total_seconds()
        case_durations.append(dur)
        
    p85_duration = np.percentile(case_durations, 85) if case_durations else 300.0
    if slow_threshold_p85 is None:
        slow_threshold_p85 = p85_duration

    labeled_rows = []
    for case_id, group in cases:
        group_sorted = group.sort_values(by='time:timestamp')
        events = group_sorted['concept:name'].tolist()
        timestamps = group_sorted['time:timestamp'].tolist()
        
        num_events = len(events)
        num_unique_telas = len(set(events))
        duration_s = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0.0
        
        has_converted = False
        if target_event:
            has_converted = target_event in events

        rework_count = num_events - num_unique_telas
        has_rework = rework_count > 0

        if num_events <= bounce_max_events and not has_converted:
            outcome_label = "Bounced"
        elif has_converted:
            outcome_label = "Converted"
        else:
            outcome_label = "Abandoned"

        if outcome_label == "Bounced":
            friction_label = "Bounce"
        elif has_rework:
            friction_label = "Loop_Friction"
        elif duration_s >= slow_threshold_p85:
            friction_label = "Bottleneck_Slow"
        elif num_unique_telas > 8:
            friction_label = "Exploratory_Wandering"
        else:
            friction_label = "Smooth_Flow"

        ux_label = f"{outcome_label}_{friction_label}"

        labeled_rows.append({
            "cd_sessao": case_id,
            "outcome_label": outcome_label,
            "friction_label": friction_label,
            "ux_label": ux_label,
            "total_eventos": num_events,
            "telas_unicas": num_unique_telas,
            "duracao_segundos": round(duration_s, 2),
            "retrabalhos_count": rework_count,
            "evento_inicial": events[0],
            "evento_final": events[-1]
        })

    return pd.DataFrame(labeled_rows)


def has_filter_arguments(args):
    """Verifica se o usuário forneceu parâmetros de filtragem/regra de rotulagem."""
    filter_attrs = [
        'start_activities', 'end_activities', 'contain', 'exclude',
        'pattern', 'not_pattern', 'trace_include', 'trace_exclude',
        'min_duration', 'max_duration', 'min_events', 'max_events'
    ]
    return any(getattr(args, attr, None) is not None for attr in filter_attrs)


def run_labeling(args):
    output_csv = resolve_output_file_path(args, args.csv, "labeling", "rotulado", "csv", explicit_override=args.output_csv)
    
    if not check_overwrite(output_csv, args):
        return

    print(f"Lendo e formatando CSV de logs: {args.csv}")
    df_raw, _, df_event_log = prepare_event_log(args.csv, args.events, temp_col=True)
    if df_event_log is None:
        return

    label_col = getattr(args, 'label_col', None) or getattr(args, 'column_name', None) or 'label'
    total_sessions = df_event_log['case:concept:name'].nunique()
    print(f"Total de sessões no log: {total_sessions}")

    if has_filter_arguments(args):
        print(f"\nAplicando regras de rotulagem baseadas em filtro para criar coluna 0/1 ('{label_col}')...")
        df_filtered_log = apply_all_filters(df_event_log, args)
        matching_session_ids = set(df_filtered_log['case:concept:name'].unique()) if df_filtered_log is not None else set()

        df_output = df_raw.copy()
        df_output[label_col] = df_output['cd_sessao'].isin(matching_session_ids).astype(int)

        match_sessions_count = len(matching_session_ids)
        no_match_sessions_count = total_sessions - match_sessions_count
        pct_match = (match_sessions_count / total_sessions * 100) if total_sessions > 0 else 0
        pct_no_match = 100.0 - pct_match

        print(f"\n--- RESUMO DA ROTULAGEM ('{label_col}') ---")
        print(f"Sessões com Match (1): {match_sessions_count} ({pct_match:.1f}%)")
        print(f"Sessões sem Match (0): {no_match_sessions_count} ({pct_no_match:.1f}%)")

    elif getattr(args, 'target_event', None):
        target_event = args.target_event.strip("'\"")
        print(f"\nAplicando rotulagem por evento alvo ('{target_event}') para criar coluna 0/1 ('{label_col}')...")
        all_activities = list(df_event_log['concept:name'].unique())
        matched_acts = [a for a in all_activities if target_event.lower() in a.lower()] if target_event not in all_activities else [target_event]
        
        if matched_acts:
            matching_session_ids = set(df_event_log[df_event_log['concept:name'].isin(matched_acts)]['case:concept:name'].unique())
        else:
            print(f"Aviso: Evento alvo '{target_event}' não foi encontrado no log.")
            matching_session_ids = set()

        df_output = df_raw.copy()
        df_output[label_col] = df_output['cd_sessao'].isin(matching_session_ids).astype(int)

        match_sessions_count = len(matching_session_ids)
        no_match_sessions_count = total_sessions - match_sessions_count
        pct_match = (match_sessions_count / total_sessions * 100) if total_sessions > 0 else 0

        print(f"\n--- RESUMO DA ROTULAGEM ('{label_col}') ---")
        print(f"Sessões com Match (1): {match_sessions_count} ({pct_match:.1f}%)")
        print(f"Sessões sem Match (0): {no_match_sessions_count} ({100.0 - pct_match:.1f}%)")

    else:
        print(f"\nNenhum filtro fornecido. Gerando rótulos comportamentais e adicionando coluna 0/1 '{label_col}' (1=Converted, 0=Outros)...")
        df_labeled = generate_heuristic_labels(df_event_log)
        converted_sessions = set(df_labeled[df_labeled['outcome_label'] == 'Converted']['cd_sessao'].unique())
        
        df_output = df_raw.copy()
        df_output[label_col] = df_output['cd_sessao'].isin(converted_sessions).astype(int)

        print("\n--- DISTRIBUIÇÃO DOS RÓTULOS COMPORTAMENTAIS GERADOS ---")
        print(df_labeled['ux_label'].value_counts().to_string())

    try:
        df_output.to_csv(output_csv, index=False)
        print(f"\nSucesso! Log de casos rotulado salvo em CSV: {os.path.abspath(output_csv)}")
    except Exception as e:
        print(f"Erro ao salvar arquivo de rótulos: {e}")
