import os
import pandas as pd
import numpy as np
import pm4py
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite

def extract_case_features(df_event_log):
    """
    Extrai matriz tabular de características (features) por sessão/caso para
    alimentar scripts externos de Machine Learning e Clusterização Não Supervisionada.
    """
    cases = df_event_log.groupby('case:concept:name')
    all_activities = sorted(df_event_log['concept:name'].unique())
    
    # 1. Bag of Activities (Frequência por Tela)
    activity_counts_per_case = []
    
    # 2. DFG Transitions (Bigramas de Navegação)
    transitions_set = set()
    case_transitions_dict = {}

    for case_id, group in cases:
        group_sorted = group.sort_values(by='time:timestamp')
        events = group_sorted['concept:name'].tolist()
        
        # Bigramas
        case_trans = []
        for i in range(len(events) - 1):
            pair = f"{events[i]}__->__{events[i+1]}"
            transitions_set.add(pair)
            case_trans.append(pair)
            
        case_transitions_dict[case_id] = case_trans

    all_transitions = sorted(list(transitions_set))

    # Construir linhas da matriz de características
    feature_rows = []
    for case_id, group in cases:
        group_sorted = group.sort_values(by='time:timestamp')
        events = group_sorted['concept:name'].tolist()
        timestamps = group_sorted['time:timestamp'].tolist()
        
        row = {"cd_sessao": case_id}
        
        # Métrica Estatísticas Gerais
        num_events = len(events)
        num_unique = len(set(events))
        duration_s = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0.0
        rework_count = num_events - num_unique
        
        row["feat_total_events"] = num_events
        row["feat_unique_activities"] = num_unique
        row["feat_unique_ratio"] = round(num_unique / num_events, 4) if num_events > 0 else 0.0
        row["feat_duration_seconds"] = round(duration_s, 2)
        row["feat_rework_count"] = rework_count

        # Counts de Atividades (Bag of Activities)
        act_counts = pd.Series(events).value_counts().to_dict()
        for act in all_activities:
            col_name = f"act_cnt__{act}"
            row[col_name] = act_counts.get(act, 0)

        # Counts de Transições (DFG Bigrams)
        trans_counts = pd.Series(case_transitions_dict.get(case_id, [])).value_counts().to_dict()
        for trans in all_transitions:
            col_name = f"trans_cnt__{trans}"
            row[col_name] = trans_counts.get(trans, 0)

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)

def run_features(args):
    output_csv = resolve_output_file_path(args, args.csv, "features", "features_ml", "csv", explicit_override=args.output_csv)
    
    if not check_overwrite(output_csv, args):
        return

    print(f"Lendo e formatando CSV de logs: {args.csv}")
    _, _, df_event_log = prepare_event_log(args.csv, args.events)
    if df_event_log is None:
        return

    print("Extraindo matriz de características (Features) por sessão para Machine Learning...")
    df_features = extract_case_features(df_event_log)

    print(f"\nMatriz de Features extraída com sucesso!")
    print(f"Total de sessões: {len(df_features)} | Total de colunas/features: {len(df_features.columns) - 1}")

    df_features.to_csv(output_csv, index=False)
    print(f"\nSucesso! Matriz de características salva em CSV: {os.path.abspath(output_csv)}")
