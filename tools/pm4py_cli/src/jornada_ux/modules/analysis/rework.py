import os
import pandas as pd
from pm4py.statistics.rework.log import get as rework_stats
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite

def run_rework(args):
    output_csv = resolve_output_file_path(args, args.csv, "analysis/rework", "rework", "csv", explicit_override=args.output_csv)
    if not check_overwrite(output_csv, args):
        return

    # Carregar e formatar
    print(f"Lendo e formatando CSV de logs: {args.csv}")
    _, _, df_event_log = prepare_event_log(args.csv, args.events)
    if df_event_log is None:
        return

    print("Calculando estatísticas nativas de rework (retrabalho/loops) por atividade...")
    try:
        rework_dict = rework_stats.apply(df_event_log)
        
        activity_counts = df_event_log['concept:name'].value_counts().to_dict()
        case_counts_per_act = df_event_log.groupby('concept:name')['case:concept:name'].nunique().to_dict()
        
        rework_rows = []
        for activity, rw_count in rework_dict.items():
            total_instances = activity_counts.get(activity, 0)
            cases_present = case_counts_per_act.get(activity, 1)
            rw_ratio_cases = (rw_count / cases_present) * 100 if cases_present > 0 else 0
            
            rework_rows.append({
                "atividade": activity,
                "casos_com_retrabalho": rw_count,
                "total_casos_presente": cases_present,
                "taxa_retrabalho_casos_pct": rw_ratio_cases,
                "total_ocorrencias_absolutas": total_instances
            })
            
        df_out = pd.DataFrame(rework_rows)
        df_out = df_out.sort_values(by="taxa_retrabalho_casos_pct", ascending=False).reset_index(drop=True)

        print(f"\n--- TOP ATIVIDADES COM MAIOR ATRITO (RETORNO/LOOPS) ---")
        for idx, row in df_out.head(10).iterrows():
            print(f" - {row['atividade']}")
            print(f"   Taxa de Retrabalho: {row['taxa_retrabalho_casos_pct']:.2f}% (em {row['casos_com_retrabalho']} de {row['total_casos_presente']} casos)")
            print(f"   Ocorrências Totais: {row['total_ocorrencias_absolutas']}")

        df_out.to_csv(output_csv, index=False)
        print(f"\nSucesso! Estatísticas de retrabalho salvas em: {os.path.abspath(output_csv)}")
    except Exception as e:
        print(f"Erro ao calcular retrabalho: {e}")
