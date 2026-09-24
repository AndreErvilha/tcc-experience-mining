import os
import pandas as pd
import pm4py
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite

def run_conformance(args):
    if not os.path.exists(args.pnml):
        print(f"Erro: Arquivo PNML '{args.pnml}' não encontrado.")
        return

    output_csv = resolve_output_file_path(args, args.csv, "analysis/conformance", "conformance", "csv", explicit_override=args.output_csv)
    if not check_overwrite(output_csv, args):
        return

    # 1. Carregar Rede de Petri
    print(f"Carregando Rede de Petri de: {args.pnml}")
    try:
        net, initial_marking, final_marking = pm4py.read_pnml(args.pnml)
    except Exception as e:
        print(f"Erro ao carregar PNML: {e}")
        return

    # 2. Carregar e formatar o CSV
    print(f"Lendo e formatando CSV de logs: {args.csv}")
    _, _, df_event_log = prepare_event_log(args.csv, args.events)
    if df_event_log is None:
        return

    # 3. Rodar Token-Based Replay
    print("Executando Token-Based Replay para cálculo de conformidade...")
    try:
        replayed_traces = pm4py.conformance_diagnostics_token_based_replay(
            df_event_log, net, initial_marking, final_marking
        )
        
        case_ids = df_event_log['case:concept:name'].unique()
        
        diagnostics = []
        for case_id, trace_res in zip(case_ids, replayed_traces):
            diagnostics.append({
                "cd_sessao": case_id,
                "fitness_score": trace_res['trace_fitness'],
                "is_conformant": trace_res['trace_is_fit'],
                "missing_tokens": trace_res['missing_tokens'],
                "consumed_tokens": trace_res['consumed_tokens'],
                "remaining_tokens": trace_res['remaining_tokens'],
                "produced_tokens": trace_res['produced_tokens']
            })
            
        df_out = pd.DataFrame(diagnostics)
        df_out.to_csv(output_csv, index=False)
        print(f"\nSucesso! Diagnóstico de conformidade salvo em: {os.path.abspath(output_csv)}")
        print(f"Média do Fitness Score: {df_out['fitness_score'].mean():.4f}")
        print(f"Jornadas perfeitamente conformes: {df_out['is_conformant'].sum()} de {len(df_out)}")
    except Exception as e:
        print(f"Erro ao rodar conformidade: {e}")
