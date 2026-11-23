# events2vector.py
import os
import argparse
import pandas as pd
import numpy as np

def format_event_token(row, include_product=True):
    event_type = str(row.get('cd_evento_tipo', '')).strip()
    page = str(row.get('page_name', '')).strip()
    btn = str(row.get('button_text', '')).strip()
    prod = str(row.get('cd_produto', '')).strip()
    
    if btn == 'nan' or not btn or btn.lower() == 'none':
        btn_str = ''
    else:
        btn_str = f'_{btn}'
        
    if event_type == 'page_view':
        token = f'view_{page}'
    elif event_type == 'click_button':
        token = f'click_{page}{btn_str}'
    else:
        token = f'{event_type}_{page}{btn_str}'
        
    if include_product and prod and prod != 'nan':
        return f'{prod}:{token}'
    return token

# Jornadas compostas (Tabela 1 revisada) e as jornadas simples que elas contem
COMPOSTAS = {
    'openfinance_fatura_paga': ['openfinance_pago', 'fatura_paga_saldo'],          # 005 = 012 -> 001
    'resgate_pago_fatura': ['cdb_resgatado', 'fatura_paga_saldo'],                 # 009 = 008 -> 001
    'fatura_paga_pix_copia_cola': ['copia_pix_paga'],                              # 010 = fatura -> 011
}
ABANDONO_COMPOSTAS = {
    'iniciou_openfinance_fatura_sem_converter': ['iniciou_openfinance_sem_converter', 'iniciou_pagar_saldo_sem_converter'],
    'iniciou_resgate_pagar_fatura_sem_converter': ['iniciou_resgate_sem_converter', 'iniciou_pagar_saldo_sem_converter'],
    'iniciou_fatura_pix_copia_cola_sem_converter': ['iniciou_copia_pix_sem_converter', 'iniciou_pagar_saldo_sem_converter'],
}


def extract_primary_label(label_series):
    """Rotulo principal da sessao a partir das flags is_* (Tabela 1 revisada).
    - Jornadas concluidas; as compostas tem prioridade e dispensam as simples que contem
      (ex.: 005 concluida -> nao entram 012 nem 001).
    - Duas ou mais jornadas sem composta que as cubra: os nomes unidos por "+", na ordem das colunas.
    - Nenhuma concluida: os abandonos, com a mesma prioridade das compostas; nenhum: "neutro".
    (Ate a versao entregue, o rotulo era a primeira flag de sucesso na ordem das colunas; ver a errata.)"""
    is_cols = [c for c in label_series.index if c.startswith('is_')]
    active = [c.replace('is_', '') for c in is_cols if label_series[c] == 1]
    success = [a for a in active if 'sem_converter' not in a]
    for comp, partes in COMPOSTAS.items():
        if comp in success:
            success = [a for a in success if a not in partes]
    if success:
        return '+'.join(success)
    for comp, partes in ABANDONO_COMPOSTAS.items():
        if comp in active:
            active = [a for a in active if a not in partes]
    if active:
        return '+'.join(active)
    return 'neutro'


def process_dataset(csv_path: str, output_csv: str = None, include_product: bool = True):
    print(f'[*] Carregando dataset de eventos: {csv_path}')
    df = pd.read_csv(csv_path)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values(by=['cd_sessao', 'timestamp'])
        
    print(f'[*] Formatando tokens para {len(df)} eventos...')
    df['event_token'] = df.apply(lambda r: format_event_token(r, include_product), axis=1)
    
    is_cols = [c for c in df.columns if c.startswith('is_')]
    
    grouped_rows = []
    print(f'[*] Agrupando por sessao (cd_sessao)...')
    for sess_id, group in df.groupby('cd_sessao', sort=False):
        tokens = group['event_token'].tolist()
        text_repr = ' '.join(tokens)
        
        label_summary = {}
        for col in is_cols:
            label_summary[col] = int(group[col].max()) if len(group[col]) > 0 else 0
            
        primary_label = extract_primary_label(pd.Series(label_summary))
        
        has_success = any(v == 1 for k, v in label_summary.items() if 'sem_converter' not in k)
        has_attrition = any(v == 1 for k, v in label_summary.items() if 'sem_converter' in k)
        
        if has_success:
            experience = 'success'
        elif has_attrition:
            experience = 'attrition'
        else:
            experience = 'none'
            
        row_dict = {
            'cd_sessao': sess_id,
            'text_representation': text_repr,
            'num_events': len(tokens),
            'primary_label': primary_label,
            'experience': experience,
            **label_summary
        }
        grouped_rows.append(row_dict)
        
    result_df = pd.DataFrame(grouped_rows)
    print(f'[+] Sucesso! {len(result_df)} sessoes geradas.')
    
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f'[+] Salvo em: {output_csv}')
        
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Converte eventos em texto por sessao.')
    parser.add_argument('--input', default='dataset/v1/total/output/labeling/total-rotulado.csv', help='CSV de eventos')
    parser.add_argument('--output', default='data/sessions_represented.csv', help='CSV de sessoes vetorizaveis')
    parser.add_argument('--no-product-prefix', action='store_true', help='Nao incluir prefixo de produto')
    args = parser.parse_args()
    
    process_dataset(args.input, args.output, include_product=not args.no_product_prefix)

if __name__ == '__main__':
    main()
