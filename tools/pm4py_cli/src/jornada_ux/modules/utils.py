import os
import sys
import pandas as pd
import pm4py

# Colunas obrigatórias e opcionais esperadas pelo CLI
REQUIRED_COLUMNS = {
    'cd_sessao':      'Identificador único da sessão/usuário (case_id). Ex: "SESS_001"',
    'cd_produto':     'Código do produto. Ex: "Global", "Cartoes"',
    'cd_sub_produto': 'Código do sub-produto. Ex: "Home", "Aquisicao_Cartao"',
    'page_name':      'Nome da tela/página. Ex: "home_banco", "form_dados_pessoais"',
    'cd_evento_tipo': 'Tipo de evento. Ex: "page_view", "click_button"',
}
OPTIONAL_COLUMNS = {
    'button_text': 'Texto do botão clicado (usado com click_button). Pode ficar vazio.',
    'timestamp':   'Data/hora do evento. Ex: "2026-08-01 10:00:00". Se ausente, usa timestamp sintético.',
}


def validate_csv_schema(csv_path, stop_on_error=False):
    """
    Valida as colunas do CSV contra o schema esperado pelo CLI.

    Imprime um relatório claro de quais colunas estão OK, quais faltam e quais são extras.
    Retorna True se todas as colunas obrigatórias estão presentes (e com dados),
    False caso contrário.

    Se stop_on_error=True, encerra o programa com sys.exit(1) em caso de erro.
    """
    if not os.path.exists(csv_path):
        print(f"\nERRO: Arquivo não encontrado: '{csv_path}'")
        if stop_on_error:
            sys.exit(1)
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"\nERRO: Não foi possível ler o CSV '{csv_path}': {e}")
        if stop_on_error:
            sys.exit(1)
        return False

    print(f"\n=== Validação do CSV: {os.path.basename(csv_path)} ===")
    print(f"Linhas carregadas: {len(df)}")
    print(f"Colunas encontradas: {list(df.columns)}\n")

    all_ok = True
    csv_columns = set(df.columns)

    # Verificar colunas obrigatórias
    print("Colunas OBRIGATÓRIAS:")
    print("-" * 60)
    for col, desc in REQUIRED_COLUMNS.items():
        if col not in csv_columns:
            print(f"  [FALTA]  {col}")
            print(f"           -> {desc}")
            all_ok = False
        elif df[col].isna().all():
            print(f"  [VAZIA]  {col} - coluna existe mas esta completamente vazia")
            print(f"           -> {desc}")
            all_ok = False
        else:
            unique_count = df[col].nunique()
            null_count = df[col].isna().sum()
            sample = str(df[col].dropna().iloc[0]) if len(df) > 0 else "(vazio)"
            print(f"  [OK]     {col}  ({unique_count} valores únicos, {null_count} nulos)")
            print(f"           Ex: \"{sample}\"")

    # Verificar colunas opcionais
    print("\nColunas OPCIONAIS:")
    print("-" * 60)
    for col, desc in OPTIONAL_COLUMNS.items():
        if col not in csv_columns:
            print(f"  [FALTA]  {col} (opcional - sera usado valor padrao)")
            print(f"           -> {desc}")
        elif df[col].isna().all() or (df[col].astype(str).str.strip() == '').all():
            print(f"  [OK]     {col} — presente mas sem dados (usará padrão)")
        else:
            non_empty = df[col].dropna().astype(str).str.strip()
            non_empty = non_empty[non_empty != '']
            sample = non_empty.iloc[0] if len(non_empty) > 0 else "(vazio)"
            has_data_pct = round(len(non_empty) / len(df) * 100, 1)
            print(f"  [OK]     {col}  ({has_data_pct}% preenchido)")
            print(f"           Ex: \"{sample}\"")

    # Verificar colunas extras (não esperadas)
    unexpected = csv_columns - set(REQUIRED_COLUMNS) - set(OPTIONAL_COLUMNS)
    if unexpected:
        print(f"\nColunas NÃO RECONHECIDAS (serão ignoradas):")
        for col in sorted(unexpected):
            print(f"  - {col}")

    # Resumo
    print(f"\n{'=' * 60}")
    if all_ok:
        print("RESULTADO: Todas as colunas obrigatórias estão presentes e com dados.")
    else:
        print("RESULTADO: Há problemas com as colunas obrigatórias (veja acima).")
        print("\nFormato esperado do CSV:")
        print("  cd_sessao,cd_produto,cd_sub_produto,page_name,cd_evento_tipo,button_text,timestamp")
        print("  SESS_001,Global,Home,home_banco,page_view,,2026-08-01 10:00:00")
        print("  SESS_001,Global,Home,home_banco,click_button,Cartões,2026-08-01 10:00:05")
    print(f"{'=' * 60}\n")

    if stop_on_error and not all_ok:
        sys.exit(1)

    return all_ok


def format_event(row):
    """
    Formata o nome do evento no padrão:
    [cd_produto:sub_produto:page_name].tipo_evento(button_text)
    """
    produto = str(row['cd_produto']).strip() if pd.notna(row.get('cd_produto')) else ""
    sub_produto = str(row['cd_sub_produto']).strip() if pd.notna(row.get('cd_sub_produto')) else ""
    page_name = str(row['page_name']).strip() if pd.notna(row.get('page_name')) else ""
    evento_tipo = str(row['cd_evento_tipo']).strip() if pd.notna(row.get('cd_evento_tipo')) else ""
    
    prefix = f"[{produto}:{sub_produto}:{page_name}]"
    
    button = row.get('button_text')
    button_str = str(button).strip() if pd.notna(button) else ""
    if button_str.lower() in ["nan", "none"]:
        button_str = ""
        
    if evento_tipo == "click_button":
        return f"{prefix}.click_button({button_str})"
    elif evento_tipo == "page_view":
        return f"{prefix}.page_view()"
    else:
        if button_str:
            return f"{prefix}.{evento_tipo}({button_str})"
        else:
            return f"{prefix}.{evento_tipo}()"

def prepare_event_log(csv_path, events, case_id='cd_sessao', activity_key='event_name', timestamp_key='timestamp', temp_col=False):
    """
    Carrega o CSV de logs, filtra pelos tipos de evento e formata no padrão PM4Py.
    Retorna uma tupla (df_raw, df_filtered, df_event_log).
    """
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo CSV '{csv_path}' não encontrado.")
        return None, None, None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Erro ao ler arquivo CSV '{csv_path}': {e}")
        return None, None, None

    df_filtered = df[df['cd_evento_tipo'].isin(events)].copy()
    if len(df_filtered) == 0:
        print(f"Erro: Nenhum evento encontrado com os tipos selecionados: {events}")
        if 'cd_evento_tipo' in df.columns:
            print(f"Tipos de eventos disponíveis no CSV: {df['cd_evento_tipo'].unique()}")
        return None, None, None

    act_col = '_event_name_temp' if temp_col else activity_key
    df_filtered[act_col] = df_filtered.apply(format_event, axis=1)

    has_real_timestamp = 'timestamp' in df_filtered.columns and pd.notna(df_filtered['timestamp']).any()
    if has_real_timestamp:
        df_filtered[timestamp_key] = pd.to_datetime(df_filtered['timestamp'])
    else:
        df_filtered[timestamp_key] = pd.to_datetime(df_filtered.groupby(case_id).cumcount(), unit='s')

    try:
        df_event_log = pm4py.format_dataframe(
            df_filtered,
            case_id=case_id,
            activity_key=act_col,
            timestamp_key=timestamp_key
        )
    except Exception as e:
        print(f"Erro ao formatar DataFrame para o padrão PM4Py: {e}")
        return None, None, None

    return df, df_filtered, df_event_log

def resolve_output_base_name(args, input_path):
    """
    Obtém o nome base para os arquivos de saída.
    """
    if getattr(args, 'output_base', None):
        return args.output_base
    if not input_path:
        return "output"
    base = os.path.splitext(os.path.basename(input_path))[0]
    if base.endswith("_petri") or base.endswith("-petri"):
        return base[:-6]
    return base

def resolve_output_file_path(args, input_path, module_subpath, output_suffix, extension, explicit_override=None):
    """
    Resolve o caminho completo de saída.
    Se explicit_override (ex: --output_csv ou --output_png) for informado, ele tem precedência.
    Caso contrário, combina output_dir / module_subpath / base_name-output_suffix.extension.
    """
    if explicit_override:
        output_dir = os.path.dirname(os.path.abspath(explicit_override))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return explicit_override

    base_dir = getattr(args, 'output_dir', None) or "output"
    base_name = resolve_output_base_name(args, input_path)
    
    subpaths = module_subpath.strip("/\\").split("/") if isinstance(module_subpath, str) else module_subpath
    target_dir = os.path.join(base_dir, *subpaths)
    os.makedirs(target_dir, exist_ok=True)

    filename = f"{base_name}-{output_suffix}.{extension}" if output_suffix else f"{base_name}.{extension}"
    return os.path.join(target_dir, filename)

def check_overwrite(file_paths, args=None):
    """
    Verifica se algum dos arquivos indicados já existe e solicita confirmação interativa ao usuário.
    Retorna True para prosseguir e False para cancelar.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    existing_files = [f for f in file_paths if f and os.path.exists(f)]
    if not existing_files:
        return True

    if args and (getattr(args, 'overwrite', False) or getattr(args, 'yes', False)):
        return True

    print(f"\nAviso: Os seguintes arquivos de saída já existem:")
    for f in existing_files:
        print(f" - {os.path.abspath(f)}")

    try:
        confirm = input("Deseja sobrescrevê-los? (y/n): ").strip().lower()
    except EOFError:
        print("Aviso: Entrada não interativa detectada. Sobrescrevendo arquivo de saída...")
        return True

    if confirm not in ['y', 'yes', 's', 'sim']:
        print("Operação cancelada pelo usuário. Nenhum arquivo foi modificado.")
        return False

    return True
