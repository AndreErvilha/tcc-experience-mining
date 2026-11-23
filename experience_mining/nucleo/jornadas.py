"""Definição única da Tabela 1 revisada, modelos e utilitários de exibição."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Jornada:
    id: str
    nome: str
    flag_sucesso: str
    flag_abandono: str
    cenario_origem: str
    composta: bool = False


JORNADAS: tuple[Jornada, ...] = (
    Jornada("001", "Pagar fatura com saldo", "fatura_paga_saldo", "iniciou_pagar_saldo_sem_converter", "001"),
    Jornada("002", "Copiar o PIX da fatura", "pix_copiado", "iniciou_copiar_pix_sem_converter", "002"),
    Jornada("003", "Parcelar a fatura", "fatura_parcelada", "iniciou_parcelamento_sem_converter", "003"),
    Jornada("004", "Histórico de faturas em PDF", "historico_visualizado", "iniciou_historico_sem_converter", "004"),
    Jornada(
        "005",
        "Trazer dinheiro (Open Finance) e pagar a fatura",
        "openfinance_fatura_paga",
        "iniciou_openfinance_fatura_sem_converter",
        "005",
        composta=True,
    ),
    Jornada("006", "PIX por chave", "pix_transferido", "iniciou_pix_sem_converter", "006"),
    Jornada("007", "Aplicar em CDB", "cdb_aplicado", "iniciou_cdb_sem_converter", "007"),
    Jornada("008", "Resgatar CDB", "cdb_resgatado", "iniciou_resgate_sem_converter", "008"),
    Jornada(
        "009",
        "Resgatar CDB e pagar a fatura",
        "resgate_pago_fatura",
        "iniciou_resgate_pagar_fatura_sem_converter",
        "009",
        composta=True,
    ),
    Jornada(
        "010",
        "Ver a fatura e pagar pelo PIX copia e cola",
        "fatura_paga_pix_copia_cola",
        "iniciou_fatura_pix_copia_cola_sem_converter",
        "010",
        composta=True,
    ),
    Jornada("011", "PIX copia e cola avulso", "copia_pix_paga", "iniciou_copia_pix_sem_converter", "010"),
    Jornada("012", "Trazer dinheiro (Open Finance) avulso", "openfinance_pago", "iniciou_openfinance_sem_converter", "005"),
)

# Compatibilidade com código legado que esperava tuplas
JORNADAS_TUPLAS = [(j.id, j.nome, j.flag_sucesso, j.flag_abandono, j.cenario_origem) for j in JORNADAS]

IDS_V10 = {
    f: j
    for j, par in {
        "001": ("fatura_paga_saldo", "iniciou_pagar_saldo_sem_converter"),
        "002": ("pix_copiado", "iniciou_copiar_pix_sem_converter"),
        "003": ("fatura_parcelada", "iniciou_parcelamento_sem_converter"),
        "004": ("historico_visualizado", "iniciou_historico_sem_converter"),
        "005": ("openfinance_pago", "iniciou_openfinance_sem_converter"),
        "006": ("pix_transferido", "iniciou_pix_sem_converter"),
        "007": ("cdb_aplicado", "iniciou_cdb_sem_converter"),
        "008": ("cdb_resgatado", "iniciou_resgate_sem_converter"),
        "009": ("resgate_pago_fatura", "iniciou_resgate_pagar_fatura_sem_converter"),
        "010": ("copia_pix_paga", "iniciou_copia_pix_sem_converter"),
    }.items()
    for f in par
}

IDS_V11 = {f: j.id for j in JORNADAS for f in (j.flag_sucesso, j.flag_abandono)}

COMPOSTAS: dict[str, list[str]] = {
    "openfinance_fatura_paga": ["openfinance_pago", "fatura_paga_saldo"],          # 005 = 012 -> 001
    "resgate_pago_fatura": ["cdb_resgatado", "fatura_paga_saldo"],                 # 009 = 008 -> 001
    "fatura_paga_pix_copia_cola": ["copia_pix_paga"],                              # 010 = fatura -> 011
}

ABANDONO_COMPOSTAS: dict[str, list[str]] = {
    "iniciou_openfinance_fatura_sem_converter": [
        "iniciou_openfinance_sem_converter",
        "iniciou_pagar_saldo_sem_converter",
    ],
    "iniciou_resgate_pagar_fatura_sem_converter": [
        "iniciou_resgate_sem_converter",
        "iniciou_pagar_saldo_sem_converter",
    ],
    "iniciou_fatura_pix_copia_cola_sem_converter": [
        "iniciou_copia_pix_sem_converter",
        "iniciou_pagar_saldo_sem_converter",
    ],
}

REPRESENTACOES_PADRAO = [
    ("tf_idf", "data/v1.0/tfidf_vectors.pkl", "TF-IDF"),
    ("minilm", "data/v1.0/minilm_embeddings.pkl", "MiniLM"),
    ("gemini", "data/v1.0/gemini_embeddings.pkl", "Gemini (3072D)"),
]

MODELOS = [
    "kmeans_tf_idf",
    "kmeans_minilm",
    "kmeans_gemini",
    "hdbscan_tf_idf",
    "hdbscan_minilm",
    "hdbscan_gemini",
]

MODELOS_COMPARATIVO = [
    "hdbscan_gemini",
    "hdbscan_minilm",
    "hdbscan_tf_idf",
    "kmeans_gemini",
    "kmeans_minilm",
    "kmeans_tf_idf",
]

NOMES_MODELOS = {
    "kmeans_tf_idf": "K-Means + TF-IDF",
    "kmeans_minilm": "K-Means + MiniLM",
    "kmeans_gemini": "K-Means + Gemini",
    "hdbscan_tf_idf": "HDBSCAN + TF-IDF",
    "hdbscan_minilm": "HDBSCAN + MiniLM",
    "hdbscan_gemini": "HDBSCAN + Gemini",
}


def tipo_rotulo(rotulo: str) -> int:
    """0 = sucesso, 1 = atrito (abandono), 2 = neutro."""
    if rotulo == "neutro":
        return 2
    return 1 if "sem_converter" in rotulo else 0


def codigo(rotulo: str, ids: dict[str, str] = IDS_V11) -> str:
    """'fatura_paga_saldo+openfinance_pago' -> '001+012 sucesso'."""
    if rotulo == "neutro":
        return "neutro"
    return "+".join(ids.get(p, p) for p in rotulo.split("+")) + (
        " atrito" if tipo_rotulo(rotulo) == 1 else " sucesso"
    )


def colunas_completas(presentes: list[str], versao: str = "v1.1") -> list[str]:
    """Todas as jornadas da versão, rótulos '+' logo após a jornada e 'neutro' por último."""
    ids = IDS_V11 if versao == "v1.1" else IDS_V10
    base = list(ids)
    extras = [r for r in presentes if r not in base and r != "neutro"]
    chave = lambda r: (codigo(r, ids).split()[0], tipo_rotulo(r))
    return sorted(base + extras, key=chave) + ["neutro"]


def legenda_jornadas() -> str:
    return "\n".join(
        "  ·  ".join(f"{j.id} {j.nome}" for j in JORNADAS[i : i + 6]) for i in (0, 6)
    )
