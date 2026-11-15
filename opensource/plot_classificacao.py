#!/usr/bin/env python3
"""Gera figuras offline para comparar modelos e estratégias de prompt na classificação.

Lê os CSVs produzidos por classificacao_experimento.py, sem alterar os resultados.
As figuras são exportadas em PNG/PDF por padrão, com um manifesto de proveniência.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")  # Execução em servidores e contêineres sem interface gráfica.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = BASE_DIR / "outputs" / "classificacao" / "resultados"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "classificacao" / "figures"
MODELS = {
    "deepseek_v4_1_flash": "DeepSeek V4.1 Flash",
    "gemini_3_8_flash": "Gemini 3.8 Flash",
}
PROMPTS = {"zero_shot": "Zero-shot", "few_shot": "Few-shot", "cot": "CoT"}
COLORS = {"deepseek_v4_1_flash": "#0072B2", "gemini_3_8_flash": "#D55E00"}
MARKERS = {"zero_shot": "o", "few_shot": "s", "cot": "^"}
CONDITIONS = [(model, prompt) for model in MODELS for prompt in PROMPTS]
METRIC_FIELDS = (
    "classificados", "total", "acuracia", "f1_macro", "custo_usd",
    "latencia_media_s", "tokens_entrada", "tokens_saida",
)


@dataclass
class Results:
    metrics: dict
    per_class: dict
    matrices: dict
    classes: list[str]
    sources: list[Path]


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: colunas ausentes: {sorted(missing)}")
        return list(reader)


def number(row, field, context, *, score=False, integer=False):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{context}: valor inválido em {field}") from error
    if not math.isfinite(value) or value < 0 or (score and value > 1):
        raise ValueError(f"{context}: valor fora do intervalo em {field}: {value}")
    if integer and not value.is_integer():
        raise ValueError(f"{context}: {field} deve ser inteiro")
    return int(value) if integer else value


def load_results(directory):
    """Valida as seis condições completas e a consistência entre métricas e matrizes."""
    sources = [directory / "metricas.csv", directory / "f1_por_classe.csv"]
    metrics = {}
    for row in read_csv(sources[0], ("modelo", "prompt", *METRIC_FIELDS)):
        key = (row["modelo"], row["prompt"])
        if key not in CONDITIONS or key in metrics:
            raise ValueError(f"metricas.csv: condição inesperada ou duplicada: {key}")
        metrics[key] = {
            field: number(row, field, str(key), score=field in ("acuracia", "f1_macro"),
                          integer=field in ("classificados", "total", "tokens_entrada", "tokens_saida"))
            for field in METRIC_FIELDS
        }
    if set(metrics) != set(CONDITIONS):
        raise ValueError("metricas.csv: são necessárias as seis condições Gemini/DeepSeek × prompts")
    totals = {row["total"] for row in metrics.values()}
    if len(totals) != 1 or not next(iter(totals)):
        raise ValueError("metricas.csv: todas as condições devem ter o mesmo total positivo")
    if any(row["classificados"] != row["total"] for row in metrics.values()):
        raise ValueError("metricas.csv: cobertura incompleta; conclua os lotes antes de comparar")

    per_class = {key: {} for key in CONDITIONS}
    for row in read_csv(sources[1], ("modelo", "prompt", "classe", "f1", "suporte")):
        key, label = (row["modelo"], row["prompt"]), row["classe"]
        if key not in per_class or label in per_class[key]:
            raise ValueError(f"f1_por_classe.csv: condição/classe inesperada ou duplicada: {key}, {label}")
        if not label.startswith("C") or not label[1:].isdigit():
            raise ValueError(f"f1_por_classe.csv: código de classe inválido: {label}")
        per_class[key][label] = {
            "f1": number(row, "f1", str(key), score=True),
            "suporte": number(row, "suporte", str(key), integer=True),
        }
    classes = sorted(per_class[CONDITIONS[0]], key=lambda label: int(label[1:]))
    if not classes:
        raise ValueError("f1_por_classe.csv: nenhuma classe encontrada")
    supports = {label: per_class[CONDITIONS[0]][label]["suporte"] for label in classes}

    matrices = {}
    for key in CONDITIONS:
        if set(per_class[key]) != set(classes):
            raise ValueError(f"f1_por_classe.csv: classes divergentes em {key}")
        if {label: per_class[key][label]["suporte"] for label in classes} != supports:
            raise ValueError(f"f1_por_classe.csv: suportes divergentes em {key}")
        path = directory / f"confusao_{key[0]}__{key[1]}.csv"
        sources.append(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        if (not rows or rows[0][1:] != classes or [row[0] for row in rows[1:] if row] != classes
                or any(len(row) != len(classes) + 1 for row in rows)):
            raise ValueError(f"{path.name}: eixos ou dimensões incompatíveis com as classes")
        matrix = np.array([
            [number({"contagem": cell}, "contagem", path.name, integer=True) for cell in row[1:]]
            for row in rows[1:]
        ], dtype=int)
        if not np.array_equal(matrix.sum(axis=1), [supports[label] for label in classes]):
            raise ValueError(f"{path.name}: suporte por classe incompatível com f1_por_classe.csv")
        if int(matrix.sum()) != metrics[key]["classificados"]:
            raise ValueError(f"{path.name}: total incompatível com metricas.csv")
        accuracy = np.trace(matrix) / matrix.sum()
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        f1 = np.divide(2 * np.diag(matrix), denominator,
                       out=np.zeros(len(classes), dtype=float), where=denominator != 0)
        if not math.isclose(accuracy, metrics[key]["acuracia"], abs_tol=0.000051):
            raise ValueError(f"{path.name}: acurácia incompatível com metricas.csv")
        if not math.isclose(float(f1.mean()), metrics[key]["f1_macro"], abs_tol=0.000051):
            raise ValueError(f"{path.name}: F1 macro incompatível com metricas.csv")
        if any(not math.isclose(value, per_class[key][label]["f1"], abs_tol=0.000051)
               for label, value in zip(classes, f1)):
            raise ValueError(f"{path.name}: F1 incompatível com f1_por_classe.csv")
        matrices[key] = matrix
    return Results(metrics, per_class, matrices, classes, sources)


def setup_axis(ax, title, ylabel):
    ax.set_title(title, loc="left", fontweight="bold", pad=14)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def grouped(ax, results, field, *, scale=1, label_format="%.2f", delta=False):
    x = np.arange(len(PROMPTS) - int(delta))
    prompts = list(PROMPTS)[int(delta):]
    for index, model in enumerate(MODELS):
        baseline = results.metrics[(model, "zero_shot")][field] if delta else 0
        values = [(results.metrics[(model, prompt)][field] - baseline) * scale for prompt in prompts]
        bars = ax.bar(x + (index - 0.5) * 0.35, values, width=0.35,
                      color=COLORS[model], label=MODELS[model])
        ax.bar_label(bars, fmt=label_format, padding=4, fontsize=9)
    ax.set_xticks(x, [PROMPTS[prompt] for prompt in prompts])


def comparison(results, fields, title, *, delta=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    fig.suptitle(title, fontweight="bold", fontsize=15)
    total = results.metrics[CONDITIONS[0]]["total"]
    fig.supxlabel(f"{total} títulos por condição · uma execução por condição", fontsize=9, color="#555555")
    for ax, (field, name, unit, scale, fmt) in zip(axes, fields):
        grouped(ax, results, field, scale=scale, label_format=fmt, delta=delta)
        setup_axis(ax, name, unit)
        if field in ("acuracia", "f1_macro") and not delta:
            ax.set_ylim(0, 105)
            ax.yaxis.set_major_formatter(PercentFormatter(100))
        elif delta:
            limit = max(abs((results.metrics[(model, prompt)][field]
                             - results.metrics[(model, "zero_shot")][field]) * scale)
                        for model in MODELS for prompt in list(PROMPTS)[1:])
            ax.set_ylim(-max(limit * 1.6, 1), max(limit * 1.6, 1))
            ax.axhline(0, color="#444444", linewidth=1)
        else:
            ax.set_ylim(0, max(ax.get_ylim()[1] * 1.16, 0.001))
    axes[0].legend(loc="upper left" if delta else "lower left", framealpha=0.95)
    return fig


def class_heatmap(results):
    matrix = np.array([[results.per_class[key][label]["f1"] for label in results.classes]
                       for key in CONDITIONS])
    fig, ax = plt.subplots(figsize=(13, 5.5), layout="constrained")
    ax.set_title("F1 por classe e condição experimental", loc="left", fontweight="bold", pad=16)
    heat = ax.imshow(matrix, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(results.classes)), [
        f"{label}\nn={results.per_class[CONDITIONS[0]][label]['suporte']}" for label in results.classes
    ])
    ax.set_yticks(range(len(CONDITIONS)), [f"{MODELS[m]} · {PROMPTS[p]}" for m, p in CONDITIONS])
    ax.set_xlabel("Classe de referência e número de títulos")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f"{value:.3f}", ha="center", va="center",
                    color="white" if value < 0.55 else "black", fontsize=10)
    fig.colorbar(heat, ax=ax, label="F1", shrink=0.85)
    return fig


def confusion_matrices(results):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    fig.suptitle("Matrizes de confusão · contagem e percentual por classe real",
                 fontsize=16, fontweight="bold")
    for ax, key in zip(axes.flat, CONDITIONS):
        matrix = results.matrices[key]
        support = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)
        heat = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"{MODELS[key[0]]}\n{PROMPTS[key[1]]}", pad=10)
        ax.set_xticks(range(len(results.classes)), results.classes, fontsize=9)
        ax.set_yticks(range(len(results.classes)), results.classes, fontsize=9)
        ax.set_xlabel("Classe predita")
        ax.set_ylabel("Classe real")
        for row, column in np.ndindex(matrix.shape):
            if matrix[row, column]:
                ax.text(column, row, f"{matrix[row, column]}\n{normalized[row, column]:.0%}",
                        ha="center", va="center", fontsize=7,
                        color="white" if normalized[row, column] > 0.5 else "black")
    colorbar = fig.colorbar(heat, ax=axes, label="Proporção na classe real", shrink=0.8)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    return fig


def quality_cost(results):
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    setup_axis(ax, "Qualidade e custo por condição", "F1 macro (eixo ampliado)")
    for (model, prompt), row in results.metrics.items():
        x, y = row["custo_usd"], row["f1_macro"]
        ax.scatter(x, y, s=105, color=COLORS[model], marker=MARKERS[prompt], zorder=3)
        # Offsets fixos evitam sobreposição dos pares próximos no conjunto original.
        offset = (9, -15) if prompt == "zero_shot" else (9, 10)
        ax.annotate(PROMPTS[prompt], (x, y), xytext=offset, textcoords="offset points", fontsize=10)
    ax.set_xlabel("Custo total registrado (US$)")
    ax.set_xlim(left=0, right=max(row["custo_usd"] for row in results.metrics.values()) * 1.3 or 0.01)
    scores = [row["f1_macro"] for row in results.metrics.values()]
    padding = max((max(scores) - min(scores)) * 0.55, 0.01)
    ax.set_ylim(max(0, min(scores) - padding), min(1, max(scores) + padding))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles = [Line2D([], [], color=COLORS[model], marker="o", linestyle="", label=name)
               for model, name in MODELS.items()]
    ax.legend(handles=handles, loc="lower right")
    return fig


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(results, output, formats, dpi):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none"})
    output.mkdir(parents=True, exist_ok=True)
    specs = [
        ("01_desempenho", lambda: comparison(results, [
            ("acuracia", "Acurácia", "Títulos classificados corretamente", 100, "%.2f"),
            ("f1_macro", "F1 macro", "Média entre classes", 100, "%.2f"),
        ], "Desempenho por modelo e estratégia de prompt")),
        ("02_variacao_vs_zero_shot", lambda: comparison(results, [
            ("acuracia", "Variação da acurácia", "Pontos percentuais vs. zero-shot", 100, "%+.2f"),
            ("f1_macro", "Variação do F1 macro", "Pontos percentuais vs. zero-shot", 100, "%+.2f"),
        ], "Efeito do prompt em relação ao zero-shot de cada modelo", delta=True)),
        ("03_custo_latencia", lambda: comparison(results, [
            ("custo_usd", "Custo total registrado", "US$ por condição", 1, "%.4f"),
            ("latencia_media_s", "Latência média por chamada", "Segundos", 1, "%.1f"),
        ], "Custo e tempo por modelo e estratégia de prompt")),
        ("04_tokens", lambda: comparison(results, [
            ("tokens_entrada", "Tokens de entrada", "Total registrado por condição", 1, "%.0f"),
            ("tokens_saida", "Tokens de saída", "Total registrado por condição", 1, "%.0f"),
        ], "Consumo de tokens · incluindo tentativas registradas")),
        ("05_f1_por_classe", lambda: class_heatmap(results)),
        ("06_matrizes_confusao", lambda: confusion_matrices(results)),
        ("07_qualidade_custo", lambda: quality_cost(results)),
    ]
    artifacts = []
    for name, plot in specs:
        fig = plot()
        try:
            for extension in formats:
                path = output / f"{name}.{extension}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                artifacts.append({"file": path.name, "sha256": sha256(path)})
        finally:
            plt.close(fig)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {"file": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                     "numpy": np.__version__},
        "parameters": {"formats": list(formats), "dpi": dpi, "models": MODELS, "prompts": PROMPTS},
        "inputs": [{"file": path.name, "sha256": sha256(path)} for path in results.sources],
        "coverage": [{"model": m, "prompt": p, "classified": results.metrics[(m, p)]["classificados"],
                      "total": results.metrics[(m, p)]["total"]} for m, p in CONDITIONS],
        "notes": [
            "Valores dos CSVs agregados; não há intervalos de confiança ou múltiplas repetições.",
            "Diferenças entre condições são descritivas; estas figuras não testam significância.",
            "Custo é o total registrado em usage.csv; latência é média por chamada, não duração total.",
            "Tokens dependem do tokenizer; tokens de raciocínio não são somados novamente à saída.",
            "Matrizes: linhas reais, colunas preditas; cores normalizadas por linha, texto com contagem e percentual.",
        ],
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Pasta dos CSVs de avaliação (padrão: %(default)s).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Pasta de figuras e manifesto (padrão: %(default)s).")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300, help="Resolução raster (padrão: %(default)s).")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi deve ser positivo")
    try:
        results = load_results(args.results_dir)
        artifacts = generate(results, args.output_dir, list(dict.fromkeys(args.formats)), args.dpi)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Erro ao gerar figuras: {error}\n")
    print(f"{len(artifacts)} figuras salvas em {args.output_dir.resolve()}")
    print("Proveniência, parâmetros e hashes: manifest.json")


if __name__ == "__main__":
    main()
