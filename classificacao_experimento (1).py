"""Experimento de classificação: 2 modelos x 3 variações de prompt, via OpenRouter.

Modelos : DeepSeek V4.1 Flash e Gemini 3.8 Flash
Prompts : zero_shot, few_shot, cot (few-shot + raciocínio), lidos dos .txt (conteúdo intocado)
Dados   : arquivo .jsonl com video_id, title e category_id (gabarito)

Uso: ajuste as constantes abaixo e execute (sem argumentos de linha de comando).
  MODE = "dry_run"  -> só gera e salva os prompts/requisições para conferência (nenhuma chamada à API)
  MODE = "run"      -> chama a API (com retomada) e, ao final, avalia
  MODE = "evaluate" -> só avalia o que já existe em disco

Requer: pip install openai tqdm      Variável de ambiente: OPENROUTER_API_KEY
"""

import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

# ----------------------------------------------------------------- CONFIGURAÇÃO
MODE = "dry_run"                     # "dry_run" | "run" | "evaluate"
DRY_RUN_PRINT_BATCHES = 1            # quantos lotes imprimir na tela por variação de prompt (dry run)

INPUT_FILE = "ground_truth_with_ids.jsonl"           # 290 títulos
ID_FIELD = "video_id"
TITLE_FIELD = "title"
GOLD_FIELD = "category_id"           # gabarito (ex.: "C2"); se ausente, a avaliação é pulada

PROMPT_FILES = {                     # nome -> arquivo
    "zero_shot": "zero_shot.txt",
    "few_shot": "few_shot.txt",
    "cot": "chain_of_thought.txt",
}

BATCH_SIZE = 10                      # 10 = lotes (protocolo dos prompts). 1 = uma chamada por título.
SHUFFLE = True                       # embaralha antes de montar os lotes (evita lotes homogêneos se o
SEED = 42                            # arquivo estiver ordenado por categoria). Mesmo lote nas 6 condições.

MODELS = {
    "deepseek_v4_1_flash": {
        "slug": "deepseek/deepseek-v4.1-flash",
        "temperature": 0.0,                # 0.0 = determinístico; None = padrão do modelo
        "reasoning": {"enabled": False},   # sem raciocínio interno; None = não envia o parâmetro
        "response_format": "json_schema",  # "json_schema" (estrito) ou "json_object"
        "max_tokens": {"zero_shot": 1000, "few_shot": 1000, "cot": 3000},
    },
    "gemini_3_8_flash": {
        "slug": "google/gemini-3.8-flash",
        "temperature": None,               # None = não envia; vale o padrão do modelo (recomendado p/ Gemini 3)
        "reasoning": {"effort": "low"},    # Gemini 3.x: níveis low/medium/high; "minimal" não existe
        "response_format": "json_schema",
        # o raciocínio interno conta no max_tokens, por isso o orçamento é maior
        "max_tokens": {"zero_shot": 4000, "few_shot": 4000, "cot": 6000},
    },
}
RUN_MODELS = None                    # None = todos; ou lista de chaves de MODELS, ex.: ["gemini_3_8_flash"]
RUN_PROMPTS = None                   # None = todos; ou lista, ex.: ["zero_shot"]
LIMIT_BATCHES = None                 # None = todos; 1 = teste de fumaça (só o 1º lote de cada condição)

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
MAX_WORKERS = 4                      # chamadas em paralelo
MAX_ATTEMPTS = 3                     # tentativas por lote (resposta inválida ou erro de API)
TIMEOUT_SECONDS = 180.0
# Se o provedor não aceitar json_schema, cai (uma vez, para o resto da execução) para json_object.
FORMAT_FALLBACK = True
OUTPUT_DIR = Path("experimento")
# -----------------------------------------------------------------------------

_lock = threading.Lock()
_fmt_state = {}                      # chave do modelo -> formato efetivamente em uso


# ------------------------------------------------------------------ PROMPTS
def load_prompt_file(path):
    """Separa o .txt em (system, user_template). Os cabeçalhos 'System Prompt:'/'User Prompt:'
    são só a moldura do arquivo; o texto dos prompts não é alterado."""
    text = Path(path).read_text(encoding="utf-8")
    m = re.match(r"\s*System Prompt:\s*\n(.*?)\n\s*User Prompt:\s*\n(.*)\Z", text, re.S)
    if not m:
        raise ValueError(f"{path}: esperado 'System Prompt:' seguido de 'User Prompt:'")
    return m.group(1).strip(), m.group(2).strip()


def extract_categories(user_template):
    m = re.search(r"## Categorias\s*\n\n(.*?)\n\n## Regras", user_template, re.S)
    if not m:
        raise ValueError("bloco '## Categorias' não encontrado")
    return m.group(1).strip()


def load_prompts():
    raw = {k: load_prompt_file(f) for k, f in PROMPT_FILES.items()}
    cats = extract_categories(raw["zero_shot"][1])
    codes = re.findall(r"^(C\d+)\s*[–-]", cats, re.M)
    if not codes:
        raise ValueError("nenhum código Cn encontrado nas categorias do zero_shot")
    system, user = raw["few_shot"]
    for k, (system, user) in raw.items():
        if user.count("{titles}") != 1:
            raise ValueError(f"{k}: o marcador {{titles}} deve aparecer exatamente uma vez")
        if "[INSERIR" in user:
            raise ValueError(f"{k}: ainda há um marcador [INSERIR ...] no prompt")
        if extract_categories(user) != cats:
            print(f"AVISO: a lista de categorias de '{k}' difere da do zero_shot.")
    return raw, codes


def render_user(user_template, batch):
    titles = "\n".join(f"{n}\t{it['title']}" for n, it in enumerate(batch, 1))  # n<TAB>título
    return user_template.replace("{titles}", titles)  # replace (não format): o prompt tem chaves JSON


def build_schema(codes, cot):
    props = {"n": {"type": "integer"}}
    if cot:
        props["raciocinio"] = {"type": "string"}      # antes de "c": o modelo raciocina e depois decide
    props["c"] = {"type": "string", "enum": codes}
    return {
        "name": "classificacao_lote",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"classificacoes": {"type": "array", "items": {
                "type": "object", "properties": props,
                "required": list(props), "additionalProperties": False}}},
            "required": ["classificacoes"],
            "additionalProperties": False,
        },
    }


# --------------------------------------------------------------------- DADOS
def load_items(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if ID_FIELD not in row or TITLE_FIELD not in row:
                raise KeyError(f"Linha {n} sem '{ID_FIELD}' ou '{TITLE_FIELD}'. Campos: {list(row)}")
            items.append({"id": str(row[ID_FIELD]),
                          "title": " ".join(str(row[TITLE_FIELD]).split()),  # sem tabs/quebras de linha
                          "gold": row.get(GOLD_FIELD)})
    if len({i["id"] for i in items}) != len(items):
        raise ValueError("IDs duplicados em " + str(path))
    return items


def make_batches(items):
    order = list(items)
    if SHUFFLE:
        random.Random(SEED).shuffle(order)
    return [order[i:i + BATCH_SIZE] for i in range(0, len(order), BATCH_SIZE)]


def write_text(path, text, encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)


def write_mapping(batches):
    """Tabela de conversão: (lote, n do prompt) <-> video_id."""
    path = OUTPUT_DIR / "mapeamento_lotes.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "n", ID_FIELD, TITLE_FIELD, "gold"])
        for b, batch in enumerate(batches):
            for n, it in enumerate(batch, 1):
                w.writerow([b, n, it["id"], it["title"], it["gold"] or ""])
    return path


def check_fewshot_leakage(prompts, items):
    titles = {it["title"] for it in items}
    for k, (_, user) in prompts.items():
        shared = [t for t in re.findall(r'Título: "(.*?)"', user) if " ".join(t.split()) in titles]
        if shared:
            print(f"AVISO: exemplos do prompt '{k}' também estão no conjunto de teste: {shared}")


# ----------------------------------------------------------------- REQUISIÇÃO
def build_request(mcfg, pkey, system, user, schema, fmt):
    rf = ({"type": "json_schema", "json_schema": schema} if fmt == "json_schema"
          else {"type": "json_object"})
    extra = {"usage": {"include": True},                   # custo em usage.cost
             "provider": {"require_parameters": True}}     # só roteia para provedores que suportam os parâmetros
    if mcfg["reasoning"] is not None:
        extra["reasoning"] = mcfg["reasoning"]
    req = {
        "model": mcfg["slug"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": mcfg["max_tokens"][pkey],
        "response_format": rf,
        "extra_body": extra,
    }
    if mcfg.get("temperature") is not None:   # None/ausente = não envia (vale o padrão do modelo)
        req["temperature"] = mcfg["temperature"]
    return req


def extract_json(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)         # remove cercas de código, se houver
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        raise ValueError("resposta sem objeto JSON")
    return json.loads(t[i:j + 1])


def validate(parsed, batch, codes, cot):
    """Exige cada posição 1..N exatamente uma vez, código válido (e raciocínio no CoT).
    Devolve linhas já convertidas para video_id."""
    rows = parsed.get("classificacoes")
    if not isinstance(rows, list):
        raise ValueError("campo 'classificacoes' ausente ou inválido")
    got = [r.get("n") for r in rows]
    expected = set(range(1, len(batch) + 1))
    if len(got) != len(set(got)):
        raise ValueError("n repetido na resposta")
    if set(got) != expected:
        raise ValueError(f"posições divergentes. faltando={expected - set(got)} extras={set(got) - expected}")
    out = []
    for r in sorted(rows, key=lambda r: r["n"]):
        if r.get("c") not in codes:
            raise ValueError(f"código inválido: {r.get('c')!r}")
        if cot and not (isinstance(r.get("raciocinio"), str) and r["raciocinio"].strip()):
            raise ValueError(f"raciocinio ausente no n={r['n']}")
        out.append({ID_FIELD: batch[r["n"] - 1]["id"], "n": r["n"], "c": r["c"],
                    "raciocinio": r.get("raciocinio", "")})
    return out


def is_format_error(msg):
    m = msg.lower()
    return any(s in m for s in ("response_format", "json_schema", "structured", "no endpoints found"))


# ------------------------------------------------------------------- DRY RUN
def dry_run(prompts, codes, batches, configs):
    n_batches = len(batches)
    for pkey, (system, user) in prompts.items():
        d = OUTPUT_DIR / "prompts" / pkey
        for b, batch in enumerate(batches):
            write_text(d / f"batch_{b:04d}.txt", f"[SYSTEM]\n{system}\n\n[USER]\n{render_user(user, batch)}")
    for mkey, pkey in configs:
        system, user = prompts[pkey]
        req = build_request(MODELS[mkey], pkey, system, render_user(user, batches[0]),
                            build_schema(codes, pkey == "cot"), MODELS[mkey]["response_format"])
        write_text(OUTPUT_DIR / "requests" / f"{mkey}__{pkey}.json",
                   json.dumps(req, ensure_ascii=False, indent=2))
    mp = write_mapping(batches)

    print(f"\nItens: {sum(len(b) for b in batches)} | lotes de {BATCH_SIZE}: {n_batches} | "
          f"condições: {len(configs)} | chamadas previstas: {n_batches * len(configs)}")
    print(f"Categorias (enum de 'c'): {codes}")
    print(f"Prompts completos em {OUTPUT_DIR / 'prompts'}/<prompt>/batch_XXXX.txt")
    print(f"Requisições do 1º lote (sem as mensagens) em {OUTPUT_DIR / 'requests'}/")
    print(f"Mapeamento lote/n <-> {ID_FIELD}: {mp}")
    for pkey, (system, user) in prompts.items():
        approx = (len(system) + len(render_user(user, batches[0]))) / 3.5
        print(f"  {pkey}: ~{approx:.0f} tokens de entrada por chamada (estimativa grosseira)")

    print("\n" + "=" * 70 + "\nPARÂMETROS POR CONDIÇÃO (exceto messages)\n" + "=" * 70)
    for mkey, pkey in configs:
        req = json.loads((OUTPUT_DIR / "requests" / f"{mkey}__{pkey}.json").read_text(encoding="utf-8"))
        req.pop("messages")
        if "json_schema" in req["response_format"]:
            req["response_format"]["json_schema"] = "<schema estrito: ver arquivo>"
        print(f"[{mkey} | {pkey}] {json.dumps(req, ensure_ascii=False)}")

    for pkey, (system, user) in prompts.items():
        for b in range(min(DRY_RUN_PRINT_BATCHES, n_batches)):
            print(f"\n{'=' * 30} {pkey.upper()} | LOTE {b} {'=' * 30}\n[SYSTEM]\n{system}\n\n"
                  f"[USER]\n{render_user(user, batches[b])}")
    print("\nMODE = 'dry_run': nenhuma chamada feita. Mude para 'run' para executar.")


# ----------------------------------------------------------------- EXECUÇÃO
def run_dir_of(mkey, pkey):
    return OUTPUT_DIR / "runs" / mkey / pkey


def run_signature(mkey, pkey, prompts, codes, batches):
    m = MODELS[mkey]
    payload = {"slug": m["slug"], "temperature": m.get("temperature"), "reasoning": m["reasoning"],
               "response_format": m["response_format"], "max_tokens": m["max_tokens"][pkey],
               "system": prompts[pkey][0], "user": prompts[pkey][1], "codes": codes,
               "batches": [[it["id"] for it in b] for b in batches]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_signature(mkey, pkey, sig):
    p = run_dir_of(mkey, pkey) / "run_signature.txt"
    if p.exists():
        if p.read_text(encoding="utf-8").strip() != sig:
            raise RuntimeError(
                f"[{mkey}/{pkey}] A configuração mudou desde a execução anterior (modelo, parâmetros, "
                f"prompt, dados, seed ou tamanho do lote). Use outro OUTPUT_DIR ou apague {p.parent}.")
    else:
        write_text(p, sig)


def append_usage(row):
    path = OUTPUT_DIR / "usage.csv"
    with _lock:
        new = not path.exists()
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["model", "prompt", "batch", "attempt", "provider", "prompt_tokens",
                            "completion_tokens", "reasoning_tokens", "cost_usd", "latency_s",
                            "finish_reason", "format", "ok"])
            w.writerow(row)


def run_task(client, mkey, pkey, b, prompts, codes, batches):
    mcfg, cot = MODELS[mkey], pkey == "cot"
    system, user_tpl = prompts[pkey]
    batch = batches[b]
    user = render_user(user_tpl, batch)
    schema = build_schema(codes, cot)
    rdir = run_dir_of(mkey, pkey)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        fmt = _fmt_state[mkey]
        text, usage, error, provider, finish, ok = "", None, None, "", "", False
        t0 = time.time()
        try:
            resp = client.chat.completions.create(**build_request(mcfg, pkey, system, user, schema, fmt))
            text = resp.choices[0].message.content or ""
            finish = resp.choices[0].finish_reason or ""
            usage = resp.usage
            provider = getattr(resp, "provider", "") or ""
            rows = validate(extract_json(text), batch, codes, cot)
            ok = True
        except Exception as e:  # noqa: BLE001 - registramos e tentamos de novo
            error = f"{type(e).__name__}: {e}"
            if fmt == "json_schema" and FORMAT_FALLBACK and is_format_error(str(e)):
                _fmt_state[mkey] = "json_object"
                error += " [formato rebaixado para json_object]"
        latency = time.time() - t0
        write_text(rdir / "raw" / f"batch_{b:04d}_attempt{attempt}.txt",
                   text + (f"\n\n[ERRO] {error}" if error else "") + f"\n[finish_reason] {finish}")
        det = getattr(usage, "completion_tokens_details", None)
        cost = getattr(usage, "cost", None)
        append_usage([mkey, pkey, b, attempt, provider, getattr(usage, "prompt_tokens", ""),
                      getattr(usage, "completion_tokens", ""), getattr(det, "reasoning_tokens", "") or "",
                      cost if cost is not None else "", f"{latency:.1f}", finish, fmt, int(ok)])
        if ok:
            write_text(rdir / "parsed" / f"batch_{b:04d}.json", json.dumps(rows, ensure_ascii=False, indent=2))
            return mkey, pkey, b, True, None
        time.sleep(2 * attempt)
    return mkey, pkey, b, False, error


def run_all(prompts, codes, batches, configs):
    from openai import OpenAI
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Defina a variável de ambiente {API_KEY_ENV}.")
    client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=TIMEOUT_SECONDS, max_retries=2)

    last = len(batches) if LIMIT_BATCHES is None else min(LIMIT_BATCHES, len(batches))
    tasks = []
    for mkey, pkey in configs:
        _fmt_state[mkey] = MODELS[mkey]["response_format"]
        check_signature(mkey, pkey, run_signature(mkey, pkey, prompts, codes, batches))
        tasks += [(mkey, pkey, b) for b in range(last)
                  if not (run_dir_of(mkey, pkey) / "parsed" / f"batch_{b:04d}.json").exists()]
    total = len(configs) * last
    print(f"Chamadas: {len(tasks)} pendentes de {total} ({total - len(tasks)} já feitas) | {MAX_WORKERS} em paralelo")
    if not tasks:
        return

    failed = []
    bar = tqdm(total=len(tasks), unit="lote", desc="Classificando") if tqdm else None
    ex = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    try:
        futs = [ex.submit(run_task, client, m, p, b, prompts, codes, batches) for m, p, b in tasks]
        for n, fut in enumerate(as_completed(futs), 1):
            m, p, b, ok, err = fut.result()
            if not ok:
                failed.append((m, p, b))
                msg = f"[{m}/{p}] lote {b} falhou após {MAX_ATTEMPTS} tentativas: {err}"
                bar.write(msg) if bar else print(msg)
            if bar:
                bar.update(1)
            elif n % 10 == 0 or n == len(tasks):
                print(f"{n}/{len(tasks)}")
    except KeyboardInterrupt:
        ex.shutdown(wait=False, cancel_futures=True)
        print("\nInterrompido. Rode de novo para retomar de onde parou.")
        raise
    finally:
        if bar:
            bar.close()
    ex.shutdown(wait=True)
    if failed:
        print(f"{len(failed)} lotes sem resultado. Rode de novo para tentar só esses.")


# ------------------------------------------------------------------ AVALIAÇÃO
def load_preds(mkey, pkey):
    preds = {}
    for f in sorted((run_dir_of(mkey, pkey) / "parsed").glob("batch_*.json")):
        b = int(f.stem.split("_")[1])
        for r in json.loads(f.read_text(encoding="utf-8")):
            preds[r[ID_FIELD]] = {**r, "batch": b}
    return preds


def prf(gold, pred, labels):
    out = {}
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[c] = (pr, rc, 2 * pr * rc / (pr + rc) if pr + rc else 0.0, tp + fn)
    return out


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def read_usage():
    agg = defaultdict(Counter)
    path = OUTPUT_DIR / "usage.csv"
    if not path.exists():
        return agg
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            a = agg[(r["model"], r["prompt"])]
            a["calls"] += 1
            a["failed"] += r["ok"] == "0"
            for fld in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                a[fld] += int(float(r[fld] or 0))
            a["cost"] += float(r["cost_usd"] or 0)
            a["latency"] += float(r["latency_s"] or 0)
    return agg


def evaluate(items, configs):
    gold = {it["id"]: it["gold"] for it in items}
    title = {it["id"]: it["title"] for it in items}
    res_dir = OUTPUT_DIR / "resultados"
    res_dir.mkdir(parents=True, exist_ok=True)
    usage = read_usage()
    per_cfg, metrics_rows, class_rows, error_rows = {}, [], [], []

    for mkey, pkey in configs:
        preds = load_preds(mkey, pkey)
        if not preds:
            continue
        per_cfg[(mkey, pkey)] = preds
        with open(run_dir_of(mkey, pkey) / "predictions.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                if it["id"] in preds:
                    p = preds[it["id"]]
                    f.write(json.dumps({ID_FIELD: it["id"], TITLE_FIELD: it["title"], "gold": it["gold"],
                                        "pred": p["c"], "batch": p["batch"], "n": p["n"],
                                        "raciocinio": p["raciocinio"]}, ensure_ascii=False) + "\n")
        u = usage[(mkey, pkey)]
        row = {"modelo": mkey, "prompt": pkey, "classificados": len(preds), "total": len(items),
               "chamadas": u["calls"], "tentativas_falhas": u["failed"],
               "tokens_entrada": u["prompt_tokens"], "tokens_saida": u["completion_tokens"],
               "tokens_raciocinio": u["reasoning_tokens"], "custo_usd": round(u["cost"], 4),
               "latencia_media_s": round(u["latency"] / u["calls"], 1) if u["calls"] else ""}
        ids = [i for i in preds if gold.get(i)]
        if ids:
            g, p = [gold[i] for i in ids], [preds[i]["c"] for i in ids]
            labels = sorted(set(g) | set(p))
            stats = prf(g, p, labels)
            row["acuracia"] = round(sum(a == b for a, b in zip(g, p)) / len(g), 4)
            row["f1_macro"] = round(sum(s[2] for s in stats.values()) / len(labels), 4)
            for c, (pr, rc, f1, sup) in stats.items():
                class_rows.append([mkey, pkey, c, round(pr, 4), round(rc, 4), round(f1, 4), sup])
            cm = Counter(zip(g, p))
            with open(res_dir / f"confusao_{mkey}__{pkey}.csv", "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["gold \\ pred"] + labels)
                for a in labels:
                    w.writerow([a] + [cm[(a, b)] for b in labels])
            for i in ids:
                if gold[i] != preds[i]["c"]:
                    error_rows.append([mkey, pkey, i, title[i], gold[i], preds[i]["c"], preds[i]["raciocinio"]])
        metrics_rows.append(row)

    if not metrics_rows:
        print("Nenhuma predição encontrada em", OUTPUT_DIR / "runs")
        return
    res_dir.mkdir(parents=True, exist_ok=True)
    cols = ["modelo", "prompt", "classificados", "total", "acuracia", "f1_macro", "chamadas",
            "tentativas_falhas", "tokens_entrada", "tokens_saida", "tokens_raciocinio",
            "custo_usd", "latencia_media_s"]
    with open(res_dir / "metricas.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(metrics_rows)
    with open(res_dir / "f1_por_classe.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "prompt", "classe", "precisao", "revocacao", "f1", "suporte"])
        w.writerows(class_rows)
    with open(res_dir / "erros.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "prompt", ID_FIELD, TITLE_FIELD, "gold", "pred", "raciocinio"])
        w.writerows(error_rows)

    # McNemar exato entre condições (mesmos itens, pareado)
    mc_rows = []
    for (ka, pa), (kb, pb) in combinations(per_cfg.items(), 2):
        common = [i for i in pa if i in pb and gold.get(i)]
        if not common:
            continue
        b = sum(pa[i]["c"] == gold[i] and pb[i]["c"] != gold[i] for i in common)
        c = sum(pa[i]["c"] != gold[i] and pb[i]["c"] == gold[i] for i in common)
        mc_rows.append([f"{ka[0]}|{ka[1]}", f"{kb[0]}|{kb[1]}", len(common), b, c, round(mcnemar_exact(b, c), 4)])
    with open(res_dir / "mcnemar.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condicao_A", "condicao_B", "n_comum", "so_A_acerta", "so_B_acerta", "p_valor"])
        w.writerows(mc_rows)

    print(f"\n{'modelo':<22}{'prompt':<11}{'cobertura':>10}{'acc':>8}{'F1-macro':>10}{'custo US$':>11}{'reas.tok':>10}")
    for r in metrics_rows:
        print(f"{r['modelo']:<22}{r['prompt']:<11}{r['classificados']:>5}/{r['total']:<4}"
              f"{r.get('acuracia', float('nan')):>8.3f}{r.get('f1_macro', float('nan')):>10.3f}"
              f"{r['custo_usd']:>11.4f}{r['tokens_raciocinio']:>10}")
    print(f"\nResultados detalhados em {res_dir}/ (metricas, f1_por_classe, confusao_*, erros, mcnemar)")
    if any(r["classificados"] < r["total"] for r in metrics_rows):
        print("AVISO: há condições com cobertura incompleta; métricas calculadas só sobre os itens classificados.")


# ----------------------------------------------------------------------- MAIN
def main():
    for stream in (sys.stdout, sys.stderr):           # console do Windows (cp1252) não imprime emojis
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    prompts, codes = load_prompts()
    items = load_items(INPUT_FILE)
    batches = make_batches(items)
    check_fewshot_leakage(prompts, items)
    if not any(it["gold"] for it in items):
        print(f"AVISO: campo '{GOLD_FIELD}' ausente: a avaliação será pulada.")
    configs = [(m, p) for m in (RUN_MODELS or MODELS) for p in (RUN_PROMPTS or prompts)]

    if MODE == "dry_run":
        dry_run(prompts, codes, batches, configs)
    elif MODE == "run":
        write_mapping(batches)
        run_all(prompts, codes, batches, configs)
        evaluate(items, configs)
    elif MODE == "evaluate":
        evaluate(items, configs)
    else:
        raise ValueError("MODE deve ser 'dry_run', 'run' ou 'evaluate'")


if __name__ == "__main__":
    main()
