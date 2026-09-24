import os
import pm4py
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite

def run_performance(args):
    output_png = resolve_output_file_path(args, args.csv, "analysis/performance", "performance", "png", explicit_override=args.output_png)
    if not check_overwrite(output_png, args):
        return

    # Carregar e formatar o CSV
    print(f"Lendo e formatando CSV de logs: {args.csv}")
    _, _, df_event_log = prepare_event_log(args.csv, args.events)
    if df_event_log is None:
        return

    # Descobrir performance DFG
    print(f"Calculando tempos e projetando performance (métrica de agregação: {args.aggregation})...")
    try:
        dfg, start_act, end_act = pm4py.discover_performance_dfg(df_event_log)
        
        pm4py.save_vis_performance_dfg(
            dfg, start_act, end_act, output_png,
            aggregation_measure=args.aggregation,
            rankdir="TB"
        )
        print(f"Sucesso! Grafo de performance vertical salvo em: {os.path.abspath(output_png)}")
    except Exception as e:
        print(f"Erro ao gerar DFG de performance: {e}")
