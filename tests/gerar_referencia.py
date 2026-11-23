"""Gera o manifesto de referência (golden master) em tests/referencia/manifesto.json."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAIDA = ROOT / "tests" / "referencia" / "manifesto.json"

ANALISE_V1_1_REL = {
    "results/metrics/comparacao_v1.0_v1.1.csv",
    "results/figures/comparacao_v1.0_v1.1.png",
    "results/dados/migracao_rotulos.csv",
    "results/dados/migracao_rotulos.png",
    "results/dados/sessoes_rotulo_revisado.csv",
}
CONFERENCIA_REL = {
    "results/dados/conferencia_rotulos.csv",
}

FIGURAS_INSTAVEIS = {
    "shap_beeswarm_panel_2x2.png",
    "beeswarm.png",
    "dfg.png",
    "rede_petri.png",
}


def sha256_texto_normalizado(path: Path) -> str:
    dados = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(dados).hexdigest()


def inspecionar_png(path: Path) -> dict:
    arr = np.ascontiguousarray(mpimg.imread(str(path)))
    estavel = path.name not in FIGURAS_INSTAVEIS
    return {
        "altura": int(arr.shape[0]),
        "largura": int(arr.shape[1]),
        "pixel_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "pixel_estavel": estavel,
    }


def inspecionar_pnml(path: Path) -> dict:
    tree = ET.parse(str(path))
    root = tree.getroot()
    places = sum(1 for el in root.iter() if el.tag.split("}")[-1] == "place")
    transitions = sum(1 for el in root.iter() if el.tag.split("}")[-1] == "transition")
    return {"places": int(places), "transitions": int(transitions)}


def classificar_grupo(rel_posix: str) -> str:
    if rel_posix in ANALISE_V1_1_REL:
        return "analise_v1_1"
    if rel_posix in CONFERENCIA_REL:
        return "conferencia_rotulos"
    return "pipeline"


def gerar_manifesto(root: Path = ROOT) -> dict:
    grupos = {
        g: {"csv_json_sha256": {}, "png_info": {}, "pnml_info": {}}
        for g in ("pipeline", "analise_v1_1", "conferencia_rotulos")
    }

    sessions_csv = root / "data" / "sessions_represented.csv"
    if not sessions_csv.exists():
        raise FileNotFoundError(f"Arquivo ausente: {sessions_csv}")
    grupos["pipeline"]["csv_json_sha256"]["data/sessions_represented.csv"] = sha256_texto_normalizado(sessions_csv)

    results_dir = root / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Pasta results/ ausente: {results_dir}")

    for path in sorted(results_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        grupo = classificar_grupo(rel)
        suf = path.suffix.lower()
        if suf in (".csv", ".json"):
            grupos[grupo]["csv_json_sha256"][rel] = sha256_texto_normalizado(path)
        elif suf == ".png":
            grupos[grupo]["png_info"][rel] = inspecionar_png(path)
        elif suf == ".pnml":
            grupos[grupo]["pnml_info"][rel] = inspecionar_pnml(path)

    return grupos


def main():
    manifesto = gerar_manifesto(ROOT)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for nome, sec in manifesto.items():
        print(
            f"[+] {nome}: {len(sec['csv_json_sha256'])} csv/json, "
            f"{len(sec['png_info'])} png, {len(sec['pnml_info'])} pnml"
        )
    print(f"[+] Manifesto salvo em: {SAIDA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
