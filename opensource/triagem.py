"""Triagem: classifica os títulos de triagem.jsonl em 4 taxonomias (uma execução por taxonomia, em
sequência), em lotes de 10, com Claude via OpenRouter. Ao final, compara as classificações obtidas
com as diferentes taxonomias (NMI, ARI e P1) para indicar qual é mais estável/adequada.

Uso: ajuste as constantes abaixo e execute o script (sem argumentos de linha de comando).
Com DRY_RUN = True, nada é enviado à API: os prompts são só gerados, salvos em .txt e
o(s) primeiro(s) de cada taxonomia é(são) impresso(s). Com DRY_RUN = False, as chamadas são feitas.

Retomada: cada lote concluído fica salvo em disco (parsed/batch_XXXX.json) e é pulado nas
próximas execuções. Interromper (Ctrl+C, queda de rede etc.) e rodar de novo não repete lotes prontos.

Requer: pip install openai   (tqdm é opcional)
Variável de ambiente: OPENROUTER_API_KEY
"""

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm é opcional
    tqdm = None


# ----------------------------------------------------------------- CONFIGURAÇÃO
DRY_RUN = False                       # True: só gera/salva os prompts. False: chama a API.
DRY_RUN_PRINT_BATCHES = 1            # quantos prompts imprimir na tela, por taxonomia, no dry run
MAX_BATCHES_PER_RUN = None           # máximo de lotes NOVOS (chamadas) por execução do script, somando
                                     # todas as taxonomias (None = todos). Rodar de novo continua.

BASE_DIR = Path(__file__).resolve().parent   # caminhos padrão são relativos a este script

# As 4 taxonomias, executadas nesta ordem. Nome -> arquivo (.jsonl com um grupo por linha,
# {"label": ..., "description": ...}; também são aceitos .json no formato {"groups": [...]}).
TAXONOMY_FILES = {
    "taxonomia_1": r"C:\Users\Thales Henrique\Documents\trabalho LLM\topic_generation\gpt6\experiment_1_original_order\taxonomia_mesclada.json",
    "taxonomia_2": r"C:\Users\Thales Henrique\Documents\trabalho LLM\topic_generation\gpt6\experiment_2_shuffled_order\taxonomia_mesclada.json",
    "taxonomia_3": BASE_DIR / "data" / "taxonomias" / "14B-taxonomia_mesclada_order.json",
    "taxonomia_4": BASE_DIR / "data" / "taxonomias" / "14B-taxonomia_mesclada_shuff.json",
}
INPUT_FILE = BASE_DIR / "data" / "triagem.jsonl"
ID_FIELD = "video_id"                # coluna com o identificador
TITLE_FIELD = "title"                # coluna com o título

MODEL = "anthropic/claude-sonnet-5"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
BATCH_SIZE = 10
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 2000
MAX_ATTEMPTS = 3
TIMEOUT_SECONDS = 120.0

NONE_LABEL = "Nenhuma das categorias"
NONE_CODE = "C0"                     # código curto da opção "nenhuma serve"
# Raciocínio no OpenRouter: {"enabled": False} desliga; None = não envia o parâmetro (padrão do modelo).
REASONING = {"enabled": False}
OUTPUT_DIR = BASE_DIR / "outputs" / "triagem_sonnet"   # Estrutura: OUTPUT_DIR/<taxonomia>/{prompts,raw,parsed,...}
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Você é um anotador especializado em classificação de textos curtos. "
    "Você recebe títulos de vídeos do YouTube, em português, sobre medicamentos para "
    "emagrecimento, e atribui a cada título exatamente uma categoria de uma lista fixa. "
    "Responda apenas com o JSON pedido."
)

USER_TEMPLATE = """\
Classifique cada título de vídeo abaixo em UMA das categorias da lista.

## Categorias

{categories}

{none_code} – {none_label}: use somente se não houver nenhum assunto identificável (por ex., hashtags soltas, nomes de produtos ou termos sem contexto) ou que não se encaixe nas categorias pré-definidas.

## Regras

1. Escolha exatamente uma categoria por título e responda com o código da categoria (por exemplo, C3).
2. Baseie-se apenas no conteúdo do título. Não invente informações.
3. Se mais de uma categoria couber, escolha a que melhor descreve o foco principal do título.
4. Avalie cada título de forma independente dos demais.
5. Devolva uma classificação para cada título, usando o número (n) correspondente.

## Títulos

Formato: n<TAB>título

{titles}

## Formato da resposta

Responda apenas com um objeto JSON no formato:
{{"classificacoes": [{{"n": 1, "c": "C3"}}, {{"n": 2, "c": "C0"}}, ...]}}
"""


# ------------------------------------------------------------------ CARREGAMENTO
def load_taxonomy(path):
    """Aceita .jsonl (um grupo por linha; ou uma linha com {"groups": [...]}) ou .json com {"groups": [...]}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Taxonomia não encontrada: {path}")
    groups = []
    with open(path, encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            data = json.load(f)
            groups = data["groups"] if isinstance(data, dict) else data
        else:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "groups" in row:
                    groups.extend(row["groups"])
                else:
                    groups.append(row)
    for g in groups:
        if "label" not in g or "description" not in g:
            raise KeyError(f"Grupo sem 'label' ou 'description' em {path}: {list(g)}")
    if not groups:
        raise ValueError(f"Taxonomia vazia: {path}")
    return [{"label": g["label"].strip(), "description": g["description"].strip()} for g in groups]


def load_items(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if ID_FIELD not in row or TITLE_FIELD not in row:
                raise KeyError(
                    f"Linha {n} sem '{ID_FIELD}' ou '{TITLE_FIELD}'. Campos: {list(row)}. "
                    "Ajuste ID_FIELD/TITLE_FIELD."
                )
            title = " ".join(str(row[TITLE_FIELD]).split())  # remove tabs/quebras de linha
            items.append({"id": str(row[ID_FIELD]), "title": title})
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("IDs duplicados em " + str(path))
    return items


def make_batches(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


# ------------------------------------------------------------ PROMPT / SCHEMA
def build_user_prompt(taxonomy, batch):
    categories = "\n".join(f"{code} – {t['label']}: {t['description']}"
                           for code, t in zip(category_codes(taxonomy), taxonomy))
    titles = "\n".join(f"{n}\t{it['title']}" for n, it in enumerate(batch, 1))
    return USER_TEMPLATE.format(categories=categories, none_code=NONE_CODE,
                                none_label=NONE_LABEL, titles=titles)


def category_codes(taxonomy):
    return [f"C{i}" for i in range(1, len(taxonomy) + 1)]


def build_schema(taxonomy):
    codes = category_codes(taxonomy) + [NONE_CODE]
    return {
        "name": "classificacao_lote",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "classificacoes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "n": {"type": "integer"},
                            "c": {"type": "string", "enum": codes},
                        },
                        "required": ["n", "c"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["classificacoes"],
            "additionalProperties": False,
        },
    }


def validate(parsed, batch, taxonomy):
    """Levanta ValueError se a resposta não cobre exatamente os itens do lote.
    Devolve {video_id: (codigo, rotulo)}."""
    code_to_label = dict(zip(category_codes(taxonomy), (t["label"] for t in taxonomy)))
    code_to_label[NONE_CODE] = NONE_LABEL
    rows = parsed.get("classificacoes")
    if not isinstance(rows, list):
        raise ValueError("campo 'classificacoes' ausente ou inválido")
    got = [r.get("n") for r in rows]
    expected = list(range(1, len(batch) + 1))
    if len(got) != len(set(got)):
        raise ValueError("n repetido na resposta")
    if set(got) != set(expected):
        raise ValueError(f"posições divergentes. faltando={set(expected) - set(got)} extras={set(got) - set(expected)}")
    for r in rows:
        if r.get("c") not in code_to_label:
            raise ValueError(f"código inválido: {r.get('c')!r}")
    return {batch[r["n"] - 1]["id"]: [r["c"], code_to_label[r["c"]]] for r in rows}


def call_model(client, user_prompt, schema):
    extra_body = {"usage": {"include": True}}  # OpenRouter devolve o custo em usage.cost
    if REASONING is not None:
        extra_body["reasoning"] = REASONING
    return client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        response_format={"type": "json_schema", "json_schema": schema},
        extra_body=extra_body,
    )


# ------------------------------------------------------------------ ARQUIVOS
def write_text(path, text):
    """Escrita atômica: grava em .tmp e renomeia, para uma interrupção nunca deixar arquivo pela metade."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run_signature(taxonomy, items):
    """Hash de tudo que muda o resultado: se mudar, os lotes antigos não valem mais."""
    payload = json.dumps({
        "system": SYSTEM_PROMPT, "template": USER_TEMPLATE, "taxonomy": taxonomy,
        "ids": [it["id"] for it in items], "titles": [it["title"] for it in items],
        "model": MODEL, "temperature": TEMPERATURE, "batch_size": BATCH_SIZE, "reasoning": REASONING,
        "none": [NONE_CODE, NONE_LABEL],
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_run_config(run_dir, taxonomy, items):
    path = run_dir / "run_signature.txt"
    sig = run_signature(taxonomy, items)
    if path.exists() and path.read_text().strip() != sig:
        raise RuntimeError(
            f"A configuração (prompt, taxonomia, modelo, dados...) mudou desde a execução anterior em {run_dir}/. "
            "Apague essa pasta (ou mude OUTPUT_DIR) para não misturar resultados de prompts diferentes."
        )
    write_text(path, sig)


def batch_is_done(parsed_path, batch):
    """Lote pronto = arquivo existe e é um JSON íntegro que cobre todos os ids do lote."""
    if not parsed_path.exists():
        return False
    try:
        labels = json.loads(parsed_path.read_text(encoding="utf-8"))
        return all(it["id"] in labels for it in batch)
    except (json.JSONDecodeError, OSError):
        return False


# ------------------------------------------------------------------ MÉTRICAS
def _comb2(n):
    return n * (n - 1) / 2


def _contingency(a, b):
    return Counter(zip(a, b)), Counter(a), Counter(b)


def nmi(a, b):
    """Informação mútua normalizada (média aritmética das entropias, como no scikit-learn)."""
    n = len(a)
    nij, na, nb = _contingency(a, b)
    ha = -sum(v / n * math.log(v / n) for v in na.values())
    hb = -sum(v / n * math.log(v / n) for v in nb.values())
    if ha == 0 and hb == 0:
        return 1.0
    mi = sum(v / n * math.log(n * v / (na[i] * nb[j])) for (i, j), v in nij.items())
    denom = (ha + hb) / 2
    return max(0.0, mi / denom) if denom > 0 else 0.0


def ari(a, b):
    """Adjusted Rand Index."""
    n = len(a)
    nij, na, nb = _contingency(a, b)
    sum_ij = sum(_comb2(v) for v in nij.values())
    sum_a = sum(_comb2(v) for v in na.values())
    sum_b = sum(_comb2(v) for v in nb.values())
    total = _comb2(n)
    if total == 0:
        return 1.0
    expected = sum_a * sum_b / total
    maximum = (sum_a + sum_b) / 2
    if maximum == expected:
        return 1.0
    return (sum_ij - expected) / (maximum - expected)


def _purity(ref, pred):
    """Pureza dos clusters de 'pred' em relação a 'ref'."""
    best = {}
    for (r, p), v in Counter(zip(ref, pred)).items():
        best[p] = max(best.get(p, 0), v)
    return sum(best.values()) / len(ref)


def p1(a, b):
    """P1 = pureza, simétrica (média das duas direções): fração dos itens que cai na classe
    majoritária do cluster correspondente na outra partição."""
    return (_purity(a, b) + _purity(b, a)) / 2


def load_predictions(tax_dir, items):
    path = tax_dir / "predictions.jsonl"
    if not path.exists():
        return None
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            preds[d[ID_FIELD]] = d["codigo"]
    if any(it["id"] not in preds for it in items):
        return None
    return preds


def analisar_estabilidade(base_dir, tax_names, items):
    """Compara as classificações das diferentes taxonomias par a par (NMI, ARI, P1).
    NMI, ARI e P1 comparam partições, então não dependem de os códigos (C1..Cn) terem o mesmo
    significado ou de as taxonomias terem o mesmo número de categorias."""
    print("\n" + "=" * 78)
    print(" ESTABILIDADE ENTRE TAXONOMIAS ")
    print("=" * 78)
    ids = [it["id"] for it in items]
    labels = {}
    for name in tax_names:
        preds = load_predictions(base_dir / name, items)
        if preds is None:
            print(f"[Aviso] {name} incompleta; ignorada na análise.")
            continue
        labels[name] = [preds[i] for i in ids]
    if len(labels) < 2:
        print("São necessárias pelo menos 2 taxonomias completas.")
        return

    # Resumo por taxonomia
    print(f"\n{'taxonomia':<14}{'categorias usadas':>19}{'% C0':>9}{'maior categoria':>17}")
    for name, lab in labels.items():
        c = Counter(lab)
        used = len([k for k in c if k != NONE_CODE])
        print(f"{name:<14}{used:>19}{100 * c.get(NONE_CODE, 0) / len(lab):>8.1f}%"
              f"{100 * max(c.values()) / len(lab):>16.1f}%")

    # Pares
    pair_rows = []
    for x, y in combinations(labels, 2):
        pair_rows.append({"taxonomia_a": x, "taxonomia_b": y, "NMI": nmi(labels[x], labels[y]),
                          "ARI": ari(labels[x], labels[y]), "P1": p1(labels[x], labels[y])})
    print(f"\n{'par':<30}{'NMI':>8}{'ARI':>8}{'P1':>8}")
    for p in pair_rows:
        print(f"{p['taxonomia_a'] + ' x ' + p['taxonomia_b']:<30}{p['NMI']:>8.4f}{p['ARI']:>8.4f}{p['P1']:>8.4f}")

    # Média de cada taxonomia contra todas as outras (quanto mais alta, mais "consensual")
    summary = []
    for name in labels:
        mine = [p for p in pair_rows if name in (p["taxonomia_a"], p["taxonomia_b"])]
        summary.append({"taxonomia": name,
                        **{m: sum(p[m] for p in mine) / len(mine) for m in ("NMI", "ARI", "P1")}})
    summary.sort(key=lambda s: s["NMI"], reverse=True)
    print(f"\nMédia de cada taxonomia contra as demais (ordenado por NMI):")
    print(f"{'taxonomia':<14}{'NMI':>8}{'ARI':>8}{'P1':>8}")
    for s in summary:
        print(f"{s['taxonomia']:<14}{s['NMI']:>8.4f}{s['ARI']:>8.4f}{s['P1']:>8.4f}")
    print(f"\nMaior concordância média com as demais: {summary[0]['taxonomia']}")

    out = base_dir / "stability.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["taxonomia_a", "taxonomia_b", "NMI", "ARI", "P1"])
        w.writeheader()
        w.writerows(pair_rows)
        w.writerows({"taxonomia_a": s["taxonomia"], "taxonomia_b": "MÉDIA_DAS_DEMAIS",
                     "NMI": s["NMI"], "ARI": s["ARI"], "P1": s["P1"]} for s in summary)
    print(f"Resultados salvos em {out}")


# ------------------------------------------------------------------ EXECUÇÃO
def consolidate(tax_dir, parsed_dir, batches):
    """Reconstrói predictions.jsonl a partir dos lotes prontos. Devolve o nº de itens escritos."""
    n_written = 0
    lines = []
    for i, batch in enumerate(batches):
        parsed_path = parsed_dir / f"batch_{i:04d}.json"
        if not batch_is_done(parsed_path, batch):
            continue
        labels = json.loads(parsed_path.read_text(encoding="utf-8"))
        for it in batch:
            lines.append(json.dumps({ID_FIELD: it["id"], TITLE_FIELD: it["title"],
                                     "codigo": labels[it["id"]][0],
                                     "categoria": labels[it["id"]][1], "batch": i}, ensure_ascii=False))
            n_written += 1
    write_text(tax_dir / "predictions.jsonl", "\n".join(lines) + ("\n" if lines else ""))
    return n_written


def main():
    # Console do Windows (cp1252) não imprime emojis: troca o que não couber por "?" em vez de dar erro.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    items = load_items(INPUT_FILE)
    batches = make_batches(items, BATCH_SIZE)
    # Carrega e valida todas as taxonomias antes de qualquer chamada (falha cedo se algum arquivo estiver errado).
    taxonomies = {name: load_taxonomy(path) for name, path in TAXONOMY_FILES.items()}

    print(f"Itens: {len(items)} | lotes de {BATCH_SIZE}: {len(batches)}")
    print(f"Taxonomias ({len(taxonomies)}): " + ", ".join(f"{n} [{len(t)} cat.]" for n, t in taxonomies.items()))

    client = None
    if not DRY_RUN:
        from openai import OpenAI
        api_key = os.environ.get(API_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"Defina a variável de ambiente {API_KEY_ENV}.")
        client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=TIMEOUT_SECONDS, max_retries=3)

    all_complete = True
    budget = MAX_BATCHES_PER_RUN  # lotes novos restantes nesta execução do script (None = ilimitado)

    for name, taxonomy in taxonomies.items():
        schema = build_schema(taxonomy)
        prompts = [build_user_prompt(taxonomy, b) for b in batches]
        tax_dir = OUTPUT_DIR / name
        raw_dir = tax_dir / "raw"
        parsed_dir = tax_dir / "parsed"
        usage_path = tax_dir / "usage.csv"

        for i, p in enumerate(prompts):
            write_text(tax_dir / "prompts" / f"batch_{i:04d}.txt", f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{p}")

        approx_tokens = (len(SYSTEM_PROMPT) + len(prompts[0])) / 3.5 if prompts else 0
        print(f"\n{'#' * 70}\n# TAXONOMIA: {name} ({len(taxonomy)} categorias + '{NONE_LABEL}') "
              f"| ~{approx_tokens:.0f} tokens de entrada/lote\n{'#' * 70}")

        if DRY_RUN:
            for i in range(min(DRY_RUN_PRINT_BATCHES, len(prompts))):
                print(f"\n[{name} | LOTE {i}]\n[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{prompts[i]}")
            print(f"\nPrompts salvos em {tax_dir / 'prompts'}/ ({len(prompts)} arquivos)")
            continue

        check_run_config(tax_dir, taxonomy, items)
        if not usage_path.exists():
            write_text(usage_path, "batch,attempt,prompt_tokens,completion_tokens,cost_usd,ok\n")

        pending = [k for k in range(len(batches))
                   if not batch_is_done(parsed_dir / f"batch_{k:04d}.json", batches[k])]
        to_run = pending if budget is None else pending[:budget]
        print(f"[{name}] lotes prontos: {len(batches) - len(pending)} | "
              f"pendentes: {len(pending)} | nesta execução: {len(to_run)}")

        failed = []
        bar = tqdm(to_run, desc=name, unit="lote") if tqdm else to_run
        log = tqdm.write if tqdm else print
        try:
            for i in bar:
                batch = batches[i]
                parsed_path = parsed_dir / f"batch_{i:04d}.json"
                ok = False
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    text, usage, error = "", None, None
                    try:
                        resp = call_model(client, prompts[i], schema)
                        text = resp.choices[0].message.content or ""
                        usage = resp.usage
                        labels = validate(json.loads(text), batch, taxonomy)
                        ok = True
                    except Exception as e:
                        error = f"{type(e).__name__}: {e}"
                    write_text(raw_dir / f"batch_{i:04d}_attempt{attempt}.txt",
                               text + (f"\n\n[ERRO] {error}" if error else ""))
                    cost = getattr(usage, "cost", None) if usage else None
                    with open(usage_path, "a", encoding="utf-8") as f:
                        f.write(f"{i},{attempt},{getattr(usage, 'prompt_tokens', '')},"
                                f"{getattr(usage, 'completion_tokens', '')},"
                                f"{cost if cost is not None else ''},{int(ok)}\n")
                    if ok:
                        # Salvo imediatamente: se interromper agora, este lote não será refeito.
                        write_text(parsed_path, json.dumps(labels, ensure_ascii=False, indent=2))
                        break
                    log(f"[{name}] lote {i}, tentativa {attempt}/{MAX_ATTEMPTS} falhou: {error}")
                    time.sleep(2 * attempt)
                if budget is not None:
                    budget -= 1
                if not ok:
                    failed.append(i)
                if tqdm:
                    bar.set_postfix(falhas=len(failed))
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário. Lotes concluídos estão salvos; rode de novo para continuar.")
            consolidate(tax_dir, parsed_dir, batches)
            return

        n_written = consolidate(tax_dir, parsed_dir, batches)
        done = sum(batch_is_done(parsed_dir / f"batch_{k:04d}.json", batches[k]) for k in range(len(batches)))
        print(f"[{name}] {n_written}/{len(items)} itens em predictions.jsonl | {done}/{len(batches)} lotes prontos.")
        if done < len(batches):
            all_complete = False
            if failed:
                print(f"[{name}] lotes sem resultado: {failed}")

    if DRY_RUN:
        print("\nDRY_RUN = True: nenhuma chamada feita. Mude para False para executar.")
        return

    if not all_complete:
        print("\nExecução incompleta. Rode de novo para continuar os lotes pendentes "
              "(os já concluídos não serão chamados novamente).")
    else:
        analisar_estabilidade(OUTPUT_DIR, list(taxonomies), items)


if __name__ == "__main__":
    main()