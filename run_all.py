# ==============================================================================
# Experience Mining Pipeline - Master Replication Script
# TCC MBA em Gestao de Negocios Digitais e IA - USP/ESALQ (2026)
# Autor: Andre Eduardo Ribeiro Ervilha
# ==============================================================================
import argparse
from pathlib import Path
import sys

from experience_mining.configuracao import (
    ErroConfiguracao,
    carregar_config_json,
    resolver_caminho_config,
)
from experience_mining.orquestracao import Pipeline

ROOT = Path(__file__).parent.resolve()
FLAGS_LEGADAS = {
    "--sem-shap",
    "--petri",
    "--reusar-agrupamentos",
    "--rotular-massa",
    "--titulos-revisados",
}


def main():
    ap = argparse.ArgumentParser(description="Reproducao do pipeline de Experience Mining guiada por configuracao JSON.")
    ap.add_argument(
        "--config",
        default=None,
        help="Caminho do arquivo JSON de configuracao da versao (padrao: variavel EM_CONFIG ou .env).",
    )
    args, extras = ap.parse_known_args()
    if extras:
        legadas = [f for f in extras if f in FLAGS_LEGADAS]
        if legadas:
            print(
                f"[!] Opcao(oes) removida(s): {', '.join(legadas)}. "
                "Declare todas as opcoes no arquivo JSON de configuracao (--config ou EM_CONFIG).",
                file=sys.stderr,
            )
        else:
            print(
                f"[!] Argumento(s) nao reconhecido(s): {' '.join(extras)}. "
                "Use apenas `--config <arquivo.json>` (ou `EM_CONFIG`).",
                file=sys.stderr,
            )
        sys.exit(2)

    try:
        cfg_path = resolver_caminho_config(args.config, root=ROOT)
        cfg = carregar_config_json(cfg_path, root=ROOT, validar_contratos=True)
    except ErroConfiguracao as exc:
        print(f"[!] Erro de configuracao: {exc}", file=sys.stderr)
        sys.exit(2)

    Pipeline(cfg).executar()


if __name__ == "__main__":
    main()
