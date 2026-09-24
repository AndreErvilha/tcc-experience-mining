import os
import pandas as pd
import pm4py
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite

def run_variants(args):
    output_csv = resolve_output_file_path(args, args.csv, "analysis/variants", "variants", "csv", explicit_override=args.output_csv)
    if not check_overwrite(output_csv, args):
        return

    # Carregar e formatar
    print(f"Lendo e formatando CSV de logs: {args.csv}")
    _, _, df_event_log = prepare_event_log(args.csv, args.events)
    if df_event_log is None:
        return

    print("Calculando variantes lógicas de jornadas...")
    try:
        variants_dict = pm4py.get_variants(df_event_log)
        
        variants_list = []
        for variant_tuple, count in variants_dict.items():
            variants_list.append({
                "variant": list(variant_tuple),
                "count": count
            })
            
        variants_df = pd.DataFrame(variants_list)
        variants_df = variants_df.sort_values(by="count", ascending=False).reset_index(drop=True)
        
        total_cases = variants_df['count'].sum()
        variants_df['percentage'] = (variants_df['count'] / total_cases) * 100

        print(f"\nTotal de variantes encontradas: {len(variants_df)} para um total de {total_cases} sessões.")
        print(f"\n--- TOP {args.top_n} VARIANTES MAIS FREQUENTES ---")
        for idx, row in variants_df.head(args.top_n).iterrows():
            print(f"\nVariante #{idx+1} | Casos: {row['count']} ({row['percentage']:.2f}%)")
            variant_path = row['variant']
            if isinstance(variant_path, list):
                for step_idx, step in enumerate(variant_path):
                    print(f"   {step_idx+1}. {step}")
            else:
                print(f"   {variant_path}")

        if output_csv:
            variants_df.to_csv(output_csv, index=False)
            print(f"\nSucesso! Estatísticas completas de variantes salvas em: {os.path.abspath(output_csv)}")
    except Exception as e:
        print(f"Erro ao obter variantes: {e}")
