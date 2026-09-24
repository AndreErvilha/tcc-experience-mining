import argparse
import sys
from jornada_ux.modules import run_discovery, run_playout, run_funnel, run_labeling, run_features, run_backbone, run_list_activities
from jornada_ux.modules.analysis import (
    run_conformance,
    run_filter,
    run_performance,
    run_rework,
    run_variants,
    run_ngrams,
    run_loops,
    run_edge_weights,
)

def add_common_output_arguments(parser):
    """Adiciona argumentos padrão de diretório e nome base de saída."""
    parser.add_argument(
        "--output_dir", "-o", type=str, default=None,
        help="Diretório de saída (ex: ./out). Se não fornecido, salva em 'output'."
    )
    parser.add_argument(
        "--output_base", type=str, default=None,
        help="Nome base dos arquivos de saída (ex: tagueamento). Se não fornecido, usa o nome do arquivo de entrada."
    )
    parser.add_argument(
        "--yes", "-y", "--overwrite", action="store_true", default=False,
        help="Sobrescreve automaticamente os arquivos de saída sem pedir confirmação."
    )

def add_filter_rules_arguments(parser):
    """Adiciona os argumentos de regras de filtragem/match ao parser."""
    parser.add_argument(
        "--start_activities", nargs="+", default=None,
        help="Lista de termos contidos nas atividades iniciais para considerar apenas estes casos."
    )
    parser.add_argument(
        "--end_activities", nargs="+", default=None,
        help="Lista de termos contidos nas atividades finais para considerar apenas estes casos."
    )
    parser.add_argument(
        "--min_duration", type=float, default=None,
        help="Duração mínima do caso em segundos."
    )
    parser.add_argument(
        "--max_duration", type=float, default=None,
        help="Duração máxima do caso em segundos."
    )
    parser.add_argument(
        "--contain", nargs="+", default=None,
        help="Mantém/rotula casos que contenham ao menos um dos termos/nomes de atividade fornecidos."
    )
    parser.add_argument(
        "--exclude", nargs="+", default=None,
        help="Remove/exclui de match casos que contenham ao menos um dos termos/nomes de atividade fornecidos."
    )
    parser.add_argument(
        "--pattern", nargs=2, action="append", default=None, metavar=("ATIV_A", "ATIV_B"),
        help="Mantém/rotula casos onde ATIV_A é eventualmente-followed por ATIV_B."
    )
    parser.add_argument(
        "--not_pattern", nargs=2, action="append", default=None, metavar=("ATIV_A", "ATIV_B"),
        help="Remove/exclui de match casos onde ATIV_A é eventualmente-followed por ATIV_B."
    )
    parser.add_argument(
        "--trace_include", nargs="+", action="append", default=None,
        help="Segmento de trace com '...' como wildcard: considera casos que casam com o padrão."
    )
    parser.add_argument(
        "--trace_exclude", nargs="+", action="append", default=None,
        help="Segmento de trace com '...' como wildcard: desconsidera casos que casam com o padrão."
    )
    parser.add_argument(
        "--min_events", type=int, default=None,
        help="Quantidade mínima de eventos por caso/sessão (filtra sessões com menos de N eventos)."
    )
    parser.add_argument(
        "--max_events", type=int, default=None,
        help="Quantidade máxima de eventos por caso/sessão (filtra sessões com mais de N eventos)."
    )

def main():
    parser = argparse.ArgumentParser(
        description="pm4py-cli: Ferramenta profissional de linha de comando para Process Mining, simulação e análise de jornadas de usuários."
    )
    
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Subcomandos principais disponíveis"
    )
    
    # ----------------------------------------------------
    # Subcomando: discovery (Mapeamento/Rastreamento)
    # ----------------------------------------------------
    parser_discovery = subparsers.add_parser(
        "discovery",
        help="Gera Redes de Petri e DFGs a partir de logs de navegação de jornada do cliente."
    )
    parser_discovery.add_argument(
        "--csv", type=str, required=True,
        help="Caminho do arquivo CSV contendo a jornada (Obrigatório)."
    )
    parser_discovery.add_argument(
        "--algo", type=str, choices=["inductive", "alpha", "heuristics"], default="inductive",
        help="Algoritmo de descoberta para a Rede de Petri (padrão: 'inductive')."
    )
    parser_discovery.add_argument(
        "--events", nargs="+", default=["page_view", "click_button"],
        help="Lista de tipos de evento (cd_evento_tipo) a serem considerados (padrão: 'page_view' e 'click_button')."
    )
    parser_discovery.add_argument(
        "--mode", type=str, choices=["petri", "dfg", "both"], default="both",
        help="O que deve ser gerado: 'petri' (Rede de Petri), 'dfg' (Directly-Follows Graph) ou 'both' (ambos, padrão)."
    )
    add_common_output_arguments(parser_discovery)

    # ----------------------------------------------------
    # Subcomando: funnel (Funil de Conversão)
    # ----------------------------------------------------
    parser_funnel = subparsers.add_parser(
        "funnel", aliases=["funil"],
        help="Gera o funil de conversão com taxas de retenção, drop-off e tempos por etapa."
    )
    parser_funnel.add_argument(
        "--csv", type=str, required=True,
        help="Caminho para o arquivo CSV contendo os logs da jornada (Obrigatório)."
    )
    parser_funnel.add_argument(
        "--steps", nargs="+", default=None,
        help="Lista ordenada das atividades do funil."
    )
    parser_funnel.add_argument(
        "--events", nargs="+", default=["page_view", "click_button"],
        help="Lista de tipos de evento a considerar (padrão: 'page_view' e 'click_button')."
    )
    parser_funnel.add_argument(
        "--output_csv", type=str, default=None,
        help="Caminho completo de saída para o CSV do funil."
    )
    parser_funnel.add_argument(
        "--output_png", type=str, default=None,
        help="Caminho completo de saída para a imagem PNG do gráfico de funil."
    )
    add_common_output_arguments(parser_funnel)

    # ----------------------------------------------------
    # Subcomando: labeling (Rotulagem e Atribuição de Colunas 0/1)
    # ----------------------------------------------------
    parser_labeling = subparsers.add_parser(
        "labeling", aliases=["rotular"],
        help="Adiciona uma coluna binária 0/1 no CSV de acordo com regras de filtro/match (contain, start/end, pattern, duration)."
    )
    parser_labeling.add_argument(
        "--csv", type=str, required=True,
        help="Caminho para o arquivo CSV contendo os logs da jornada (Obrigatório)."
    )
    parser_labeling.add_argument(
        "--label_col", type=str, default="label",
        help="Nome da coluna de rótulo binário (0 ou 1) criada no CSV de saída (padrão: 'label')."
    )
    parser_labeling.add_argument(
        "--target_event", type=str, default=None,
        help="Nome do evento final de conversão alvo (ex: 'checkout_success'). Se omitido, infere euristicamente."
    )
    parser_labeling.add_argument(
        "--events", nargs="+", default=["page_view", "click_button"],
        help="Lista de tipos de evento a considerar (padrão: 'page_view' e 'click_button')."
    )
    parser_labeling.add_argument(
        "--output_csv", type=str, default=None,
        help="Caminho completo de saída para o CSV com casos rotulados."
    )
    add_filter_rules_arguments(parser_labeling)
    add_common_output_arguments(parser_labeling)

    # ----------------------------------------------------
    # Subcomando: features (Vetorização para ML)
    # ----------------------------------------------------
    parser_features = subparsers.add_parser(
        "features",
        help="Extrai matriz tabular de características por sessão (Bag of Words, DFG Bigrams) para scripts de ML."
    )
    parser_features.add_argument(
        "--csv", type=str, required=True,
        help="Caminho para o arquivo CSV contendo os logs da jornada (Obrigatório)."
    )
    parser_features.add_argument(
        "--events", nargs="+", default=["page_view", "click_button"],
        help="Lista de tipos de evento a considerar (padrão: 'page_view' e 'click_button')."
    )
    parser_features.add_argument(
        "--output_csv", type=str, default=None,
        help="Caminho completo de saída para a matriz de características CSV."
    )
    add_common_output_arguments(parser_features)

    # ----------------------------------------------------
    # Subcomando: playout (Simulação)
    # ----------------------------------------------------
    parser_playout = subparsers.add_parser(
        "playout",
        help="Simula novas jornadas de usuário a partir de um modelo estrutural (Playout estocástico)."
    )
    parser_playout.add_argument(
        "--pnml", type=str, required=True,
        help="Caminho para o arquivo PNML da Rede de Petri (Obrigatório)."
    )
    parser_playout.add_argument(
        "--csv", type=str, required=True,
        help="Caminho para o arquivo CSV de referência contendo os logs para cálculo de frequências (Obrigatório)."
    )
    parser_playout.add_argument(
        "--num_traces", type=int, default=100,
        help="Quantidade de jornadas de usuários a serem simuladas (padrão: 100)."
    )
    parser_playout.add_argument(
        "--seed", type=int, default=42,
        help="Semente para garantir a simulação idêntica e reprodutível (padrão: 42)."
    )
    parser_playout.add_argument(
        "--events", nargs="+", default=["page_view", "click_button"],
        help="Lista de tipos de evento a considerar no log de referência (padrão: 'page_view' e 'click_button')."
    )
    parser_playout.add_argument(
        "--output_csv", type=str, default=None,
        help="Caminho completo do arquivo CSV de saída (Sobrescreve a estrutura padrão de pastas)."
    )
    parser_playout.add_argument(
        "--start_session_at", type=int, default=1,
        help="Inicia código de sessao de cada jornada (ex: SESS_001) com uma contagem pre configurada (padrao: 1)"
    )
    add_common_output_arguments(parser_playout)

    # ----------------------------------------------------
    # Subcomando: backbone (Espinha Dorsal)
    # ----------------------------------------------------
    parser_backbone = subparsers.add_parser(
        "backbone",
        help="Extrai a Rede de Petri simplificada com apenas as transições de maior peso (espinha dorsal do processo)."
    )
    parser_backbone.add_argument(
        "--csv", type=str, required=True,
        help="Caminho do arquivo CSV contendo a jornada (Obrigatório)."
    )
    parser_backbone.add_argument(
        "--events", nargs="+", default=["page_view", "click_button"],
        help="Lista de tipos de evento (cd_evento_tipo) a serem considerados (padrão: 'page_view' e 'click_button')."
    )
    filter_group = parser_backbone.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--top_pct", type=float, default=None,
        help="Manter arestas acumuladas até X%% do peso total do DFG (ex: 50)."
    )
    filter_group.add_argument(
        "--min_weight", type=int, default=None,
        help="Manter apenas arestas com ao menos N ocorrências no DFG."
    )
    filter_group.add_argument(
        "--top_n", type=int, default=None,
        help="Manter apenas as N arestas de maior peso do DFG."
    )
    parser_backbone.add_argument(
        "--algo", type=str, choices=["dfg_convert", "inductive", "alpha", "heuristics"], default="dfg_convert",
        help="Algoritmo de descoberta da Rede de Petri (padrão: 'dfg_convert'). "
             "'alpha' costuma ser melhor com poucos dados."
    )
    add_common_output_arguments(parser_backbone)

    # ----------------------------------------------------
    # Subcomando: list-activities (Listar Atividades Distintas)
    # ----------------------------------------------------
    parser_list_activities = subparsers.add_parser(
        "list-activities",
        help="Lista todos os nomes exatos de atividade de um CSV, para uso em comandos de filtro."
    )
    parser_list_activities.add_argument(
        "--csv", type=str, required=True,
        help="Caminho do arquivo CSV contendo a jornada (Obrigatório)."
    )
    parser_list_activities.add_argument(
        "--events", nargs="+", default=["page_view", "click_button"],
        help="Lista de tipos de evento (cd_evento_tipo) a serem considerados (padrão: 'page_view' e 'click_button')."
    )

    # ----------------------------------------------------
    # Subcomando Pai: analysis / analise (Nested Subparsers)
    # ----------------------------------------------------
    parser_analysis = subparsers.add_parser(
        "analysis", aliases=["analise"],
        help="Sub-suíte de ferramentas para análise, conformidade, tempos, variantes e filtros de jornadas."
    )
    
    analysis_subparsers = parser_analysis.add_subparsers(
        dest="analysis_command", required=True, help="Comandos de análise disponíveis"
    )
    
    # Sub-subcomando: conformance
    parser_conformance = analysis_subparsers.add_parser(
        "conformance",
        help="Mapeia a conformidade e o Fitness Score de cada sessão em relação a uma Rede de Petri semente."
    )
    parser_conformance.add_argument(
        "--csv", type=str, required=True,
        help="Caminho para o arquivo CSV contendo os logs a serem avaliados (Obrigatório)."
    )
    parser_conformance.add_argument(
        "--pnml", type=str, required=True,
        help="Caminho para o arquivo PNML da Rede de Petri de referência/sucesso (Obrigatório)."
    )
    parser_conformance.add_argument(
        "--events", nargs="+", default=["page_view", "click_button"],
        help="Lista de tipos de evento a considerar (padrão: 'page_view' e 'click_button')."
    )
    parser_conformance.add_argument(
        "--output_csv", type=str, default=None,
        help="Caminho completo de saída para o CSV de conformidade (Sobrescreve a estrutura padrão de pastas)."
    )
    add_common_output_arguments(parser_conformance)

    # Sub-subcomando: performance
    parser_performance = analysis_subparsers.add_parser(
        "performance",
        help="Calcula tempos entre cliques/telas e gera o mapa de calor de desempenho (Performance DFG)."
    )
    parser_performance.add_argument(
        "--csv", type=str, required=True,
        help="Caminho para o arquivo CSV contendo os logs com timestamps (Obrigatório)."
    )
    parser_performance.add_argument(
        "--events", nargs="+", default=["page_view", "click_button"],
        help="Lista de tipos de evento a considerar (padrão: 'page_view' e 'click_button')."
    )
    parser_performance.add_argument(
        "--aggregation", type=str, choices=["mean", "median", "min", "max", "stdev"], default="mean",
        help="Métrica estatística para exibir nas setas do DFG de desempenho (padrão: 'mean')."
    )
    parser_performance.add_argument(
        "--output_png", type=str, default=None,
        help="Caminho completo de saída para a imagem PNG do DFG de desempenho (Sobrescreve a estrutura padrão de pastas)."
    )
    add_common_output_arguments(parser_performance)

    # Sub-subcomando: variants
    parser_variants = analysis_subparsers.add_parser(
        "variants",
        help="Mapeia e ranqueia as rotas/variantes de navegação mais tomadas pelos usuários."
    )
    parser_variants.add_argument(
        "--csv", type=str, required=True,
        help="Caminho para o arquivo CSV contendo os logs (Obrigatório)."
    )
    parser_variants.add_argument(
        "--events", nargs="+", default=["page_view", "click_button"],
        help="Lista de tipos de evento a considerar (padrão: 'page_view' e 'click_button')."
    )
    parser_variants.add_argument(
        "--top_n", type=int, default=10,
        help="Quantidade de variantes mais frequentes a exibir no terminal (padrão: 10)."
    )
    parser_variants.add_argument(
        "--output_csv", type=str, default=None,
        help="Caminho completo para salvar o CSV de variantes (Sobrescreve a estrutura padrão de pastas)."
    )
    add_common_output_arguments(parser_variants)

    # Sub-subcomando: filter
    parser_filter = analysis_subparsers.add_parser(
        "filter",
        help="Limpa e filtra o dataset de logs de acordo com regras de início, fim ou duração."
    )
    parser_filter.add_argument(
        "--csv", type=str, required=True,
        help="Caminho para o arquivo CSV contendo os logs (Obrigatório)."
    )
    parser_filter.add_argument(
        "--events", nargs="+", default=["page_view", "click_button"],
        help="Lista de tipos de evento a considerar (padrão: 'page_view' e 'click_button')."
    )
    parser_filter.add_argument(
        "--start_activities", nargs="+", default=None,
        help="Lista de termos contidos nas atividades iniciais para manter apenas estes casos."
    )
    parser_filter.add_argument(
        "--end_activities", nargs="+", default=None,
        help="Lista de termos contidos nas atividades finais para manter apenas estes casos."
    )
    parser_filter.add_argument(
        "--min_duration", type=float, default=None,
        help="Duração mínima do caso em segundos."
    )
    parser_filter.add_argument(
        "--max_duration", type=float, default=None,
        help="Duração máxima do caso em segundos."
    )
    parser_filter.add_argument(
        "--output_csv", type=str, default=None,
        help="Caminho completo de saída para o CSV resultante limpo (Sobrescreve a estrutura padrão de pastas)."
    )
    parser_filter.add_argument(
        "--contain", nargs="+", default=None,
        help="Match exato: mantém apenas casos que contenham ao menos um dos nomes de atividade fornecidos."
    )
    parser_filter.add_argument(
        "--exclude", nargs="+", default=None,
        help="Match exato: remove casos que contenham ao menos um dos nomes de atividade fornecidos."
    )
    parser_filter.add_argument(
        "--pattern", nargs=2, action="append", default=None, metavar=("ATIV_A", "ATIV_B"),
        help="Match exato: mantém casos onde ATIV_A é eventually-followed por ATIV_B. "
             "Pode ser usado múltiplas vezes para múltiplas relações."
    )
    parser_filter.add_argument(
        "--not_pattern", nargs=2, action="append", default=None, metavar=("ATIV_A", "ATIV_B"),
        help="Match exato: remove casos onde ATIV_A é eventually-followed por ATIV_B. "
             "Pode ser usado múltiplas vezes para múltiplas relações."
    )
    parser_filter.add_argument(
        "--trace_include", nargs="+", action="append", default=None,
        help="Segmento de trace com '...' como wildcard: mantém casos que casam com o padrão. "
             "Ex: --trace_include ... ATIV_A ... ATIV_B ... (A aparece antes de B). "
             "Pode ser usado múltiplas vezes."
    )
    parser_filter.add_argument(
        "--trace_exclude", nargs="+", action="append", default=None,
        help="Segmento de trace com '...' como wildcard: remove casos que casam com o padrão. "
             "Ex: --trace_exclude ... ATIV_A ... ATIV_B ... ATIV_A (A, depois B, depois A de novo). "
             "Pode ser usado múltiplas vezes."
    )
    parser_filter.add_argument(
        "--min_events", type=int, default=None,
        help="Quantidade mínima de eventos por caso/sessão (filtra sessões com menos de N eventos)."
    )
    parser_filter.add_argument(
        "--max_events", type=int, default=None,
        help="Quantidade máxima de eventos por caso/sessão (filtra sessões com mais de N eventos)."
    )
    add_common_output_arguments(parser_filter)

    # Sub-subcomando: rework
    parser_rework = analysis_subparsers.add_parser(
        "rework",
        help="Analisa loops e repetições calculando estatísticas de retrabalho por tela."
    )
    parser_rework.add_argument(
        "--csv", type=str, required=True,
        help="Caminho para o arquivo CSV contendo os logs (Obrigatório)."
    )
    parser_rework.add_argument(
        "--events", nargs="+", default=["page_view", "click_button"],
        help="Lista de tipos de evento a considerar (padrão: 'page_view' e 'click_button')."
    )
    parser_rework.add_argument(
        "--output_csv", type=str, default=None,
        help="Caminho completo de saída para o CSV de estatísticas de loops (Sobrescreve a estrutura padrão de pastas)."
    )
    add_common_output_arguments(parser_rework)

    # Sub-subcomando: ngrams
    parser_ngrams = analysis_subparsers.add_parser(
        "ngrams",
        help="Extrai e ranqueia os N-gramas (sequências contíguas de passos) mais frequentes entre as sessões."
    )
    parser_ngrams.add_argument("--csv", type=str, required=True, help="Caminho do CSV de logs.")
    parser_ngrams.add_argument("--events", nargs="+", default=["page_view", "click_button"], help="Tipos de eventos.")
    parser_ngrams.add_argument("--n", type=int, default=2, help="Tamanho do N-grama (padrão: 2, i.e. bigramas).")
    parser_ngrams.add_argument("--top_k", type=int, default=20, help="Quantos N-gramas exibir no terminal (padrão: 20).")
    parser_ngrams.add_argument("--min_count", type=int, default=1, help="Frequência mínima para incluir no CSV (padrão: 1).")
    parser_ngrams.add_argument("--highlight_ngrams", action="store_true", default=False, help="Gera um PNG individual por n-grama com arestas destacadas no grafo completo.")
    parser_ngrams.add_argument("--output_csv", type=str, default=None, help="Caminho do CSV de saída.")
    add_common_output_arguments(parser_ngrams)

    # Sub-subcomando: loops
    parser_loops = analysis_subparsers.add_parser(
        "loops",
        help="Detecta e ranqueia atividades com loops (repetições na mesma sessão), revelando pontos de fricção."
    )
    parser_loops.add_argument("--csv", type=str, required=True, help="Caminho do CSV de logs.")
    parser_loops.add_argument("--events", nargs="+", default=["page_view", "click_button"], help="Tipos de eventos.")
    parser_loops.add_argument("--min_repetitions", type=int, default=2, help="Mínimo de ocorrências para contar como loop (padrão: 2).")
    parser_loops.add_argument("--top_k", type=int, default=20, help="Quantas atividades exibir no terminal (padrão: 20).")
    parser_loops.add_argument("--output_csv", type=str, default=None, help="Caminho do CSV de saída.")
    add_common_output_arguments(parser_loops)

    # Sub-subcomando: edge-weights
    parser_edge_weights = analysis_subparsers.add_parser(
        "edge-weights",
        help="Distribuição de pesos das arestas do DFG com detecção de cotovelo, para escolha de threshold no backbone."
    )
    parser_edge_weights.add_argument("--csv", type=str, required=True, help="Caminho do CSV de logs.")
    parser_edge_weights.add_argument("--events", nargs="+", default=["page_view", "click_button"], help="Tipos de eventos.")
    parser_edge_weights.add_argument("--top_n", type=int, default=None, help="Exibir apenas as N arestas de maior peso no terminal.")
    parser_edge_weights.add_argument("--output_csv", type=str, default=None, help="Caminho do CSV de saída.")
    parser_edge_weights.add_argument("--dfg", action="store_true", default=False, help="Gerar PNG + .dfg do DFG completo.")
    add_common_output_arguments(parser_edge_weights)

    # Sub-subcomando: funnel (sob analysis também)
    parser_funnel_sub = analysis_subparsers.add_parser("funnel", aliases=["funil"], help="Gera o funil de conversão da jornada.")
    parser_funnel_sub.add_argument("--csv", type=str, required=True, help="Caminho do CSV de logs.")
    parser_funnel_sub.add_argument("--steps", nargs="+", default=None, help="Lista ordenada dos passos do funil.")
    parser_funnel_sub.add_argument("--events", nargs="+", default=["page_view", "click_button"], help="Tipos de eventos.")
    parser_funnel_sub.add_argument("--output_csv", type=str, default=None, help="Caminho do CSV de saída.")
    parser_funnel_sub.add_argument("--output_png", type=str, default=None, help="Caminho do PNG de saída.")
    add_common_output_arguments(parser_funnel_sub)

    # Sub-subcomando: labeling (sob analysis também)
    parser_labeling_sub = analysis_subparsers.add_parser("labeling", aliases=["rotular"], help="Adiciona uma coluna binária 0/1 no CSV com base em regras de filtro/match.")
    parser_labeling_sub.add_argument("--csv", type=str, required=True, help="Caminho do CSV de logs.")
    parser_labeling_sub.add_argument("--label_col", type=str, default="label", help="Nome da coluna de rótulo binário (0 ou 1) criada no CSV (padrão: 'label').")
    parser_labeling_sub.add_argument("--target_event", type=str, default=None, help="Evento final de conversão.")
    parser_labeling_sub.add_argument("--events", nargs="+", default=["page_view", "click_button"], help="Tipos de eventos.")
    parser_labeling_sub.add_argument("--output_csv", type=str, default=None, help="Caminho do CSV de saída.")
    add_filter_rules_arguments(parser_labeling_sub)
    add_common_output_arguments(parser_labeling_sub)

    # Sub-subcomando: features (sob analysis também)
    parser_features_sub = analysis_subparsers.add_parser("features", help="Extrai recursos (Features) tabulares para ML.")
    parser_features_sub.add_argument("--csv", type=str, required=True, help="Caminho do CSV de logs.")
    parser_features_sub.add_argument("--events", nargs="+", default=["page_view", "click_button"], help="Tipos de eventos.")
    parser_features_sub.add_argument("--output_csv", type=str, default=None, help="Caminho do CSV de saída.")
    add_common_output_arguments(parser_features_sub)

    args = parser.parse_args()

    # Roteamento de Comandos
    if args.command == "discovery":
        run_discovery(args)
    elif args.command in ["funnel", "funil"]:
        run_funnel(args)
    elif args.command in ["labeling", "rotular"]:
        run_labeling(args)
    elif args.command == "features":
        run_features(args)
    elif args.command == "playout":
        run_playout(args)
    elif args.command == "backbone":
        run_backbone(args)
    elif args.command == "list-activities":
        run_list_activities(args)
    elif args.command in ["analysis", "analise"]:
        if args.analysis_command == "conformance":
            run_conformance(args)
        elif args.analysis_command == "performance":
            run_performance(args)
        elif args.analysis_command == "variants":
            run_variants(args)
        elif args.analysis_command == "filter":
            run_filter(args)
        elif args.analysis_command == "rework":
            run_rework(args)
        elif args.analysis_command == "ngrams":
            run_ngrams(args)
        elif args.analysis_command == "loops":
            run_loops(args)
        elif args.analysis_command == "edge-weights":
            run_edge_weights(args)
        elif args.analysis_command in ["funnel", "funil"]:
            run_funnel(args)
        elif args.analysis_command in ["labeling", "rotular"]:
            run_labeling(args)
        elif args.analysis_command == "features":
            run_features(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

