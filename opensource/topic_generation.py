#!/usr/bin/env python3
"""Experimento compartilhado pelo notebook OpenAI e pela execução via vLLM.

Os dois provedores usam Chat Completions, reasoning_effort e o mesmo protocolo.
Os parâmetros experimentais ficam em experiment_config.json.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

PROMPT_VERSION = "topic-generation-v5-first-topic"
DEFAULT_CONFIG = Path(__file__).with_name("experiment_config.json")
SYSTEM_PROMPT = """
Você é um pesquisador especializado em análise temática. Sua tarefa é identificar novos tópicos em títulos de vídeos do YouTube
sobre medicamentos para emagrecimento. Um tópico representa um assunto temático recorrente, e não apenas uma palavra ou medicamento mencionado.

Você receberá:
1. Os tópicos identificados nos lotes anteriores, um por linha, no formato - rótulo: descrição.
2. Um novo lote de títulos, um por linha, no formato video_id<TAB>título (separados por tabulação).

Regras:
- Considere os tópicos existentes antes de propor um novo.
- Crie tópicos somente quando houver um tema relevante que ainda não seja contemplado pelos tópicos existentes.
- Não crie tópicos apenas por diferenças de palavras, medicamentos ou estilo dos títulos.
- Evite tópicos redundantes ou excessivamente específicos.
- Não altere, renomeie, combine ou exclua tópicos existentes.
- Considere o assunto do título, não a veracidade de suas afirmações.
- Quando a taxonomia estiver vazia, identifique pelo menos um tópico sustentado pelos títulos do lote.
- Quando já houver tópicos, retorne uma lista vazia se nenhum tema novo for necessário.
- Tópicos apresentados na lista TÓPICOS IDENTIFICADOS ATÉ AGORA já existem. Não repita seus rótulos.
- Uma resposta com new_topics: [] está correta somente quando já houver tópicos e os temas do lote já estiverem cobertos.
- evidence_video_ids deve conter SOMENTE identificadores exatos (primeira coluna antes do TAB), nunca títulos, nomes de tópicos ou nomes de medicamentos.
- Não repita IDs de evidência dentro do mesmo tópico. Selecione apenas IDs do lote atual que realmente sustentem o tema.
- Os nomes e descrições devem estar em português.

Para cada tópico novo, forneça:
- label: nome curto e conciso.
- description: descrição em uma frase.
- evidence_video_ids: exemplos de IDs de um ou mais títulos do lote que justificam a criação do tópico.

Retorne somente um objeto JSON com a propriedade new_topics contendo os tópicos novos.
""".strip()

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "new_topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "evidence_video_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label", "description", "evidence_video_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["new_topics"],
    "additionalProperties": False,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_json(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_titles(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido na linha {line_no}: {exc}") from exc
            if not isinstance(raw, dict) or "video_id" not in raw or "title" not in raw:
                raise ValueError(f"Linha {line_no} precisa conter video_id e title")
            video_id = str(raw["video_id"]).strip()
            title = " ".join(str(raw["title"]).split())
            if not video_id or not title or re.search(r"\s", video_id):
                raise ValueError(f"ID ou título inválido na linha {line_no}")
            if video_id in seen:
                raise ValueError(f"video_id duplicado na linha {line_no}: {video_id}")
            seen.add(video_id)
            records.append({"video_id": video_id, "title": title})
    if not records:
        raise ValueError("Arquivo JSONL não contém títulos válidos")
    return records


def build_prompt(existing: list[dict], batch: list[dict]) -> str:
    topics_text = "\n".join(f"- {t.get('topic_id', '?')} | {t['label']}: {t['description']}" for t in existing) or "(nenhum tópico ainda)"
    titles_text = "\n".join(f"{r['video_id']}\t{r['title']}" for r in batch)
    topic_rule = (
        "Não recrie tópicos já listados; quando tudo estiver coberto, retorne {\"new_topics\": []}."
        if existing else
        "A taxonomia está vazia: retorne pelo menos um tópico sustentado pelos títulos deste lote."
    )
    return (
        f"TÓPICOS IDENTIFICADOS ATÉ AGORA ({len(existing)}):\n{topics_text}\n\n"
        f"NOVO LOTE DE TÍTULOS (video_id<TAB>título):\n{titles_text}\n\n"
        "Identifique somente tópicos novos que sejam necessários. "
        "A primeira coluna de cada linha é o video_id. "
        "Use APENAS esses video_id exatos em evidence_video_ids (sem repetir IDs). "
        f"{topic_rule}"
    )


def schema_for_batch(batch: list[dict], require_topic: bool = False) -> dict[str, Any]:
    """Restringe IDs a valores que aparecem no lote atual via JSON Schema enum."""
    schema = copy.deepcopy(OUTPUT_SCHEMA)
    allowed_ids = [record["video_id"] for record in batch]
    schema["properties"]["new_topics"]["items"]["properties"]["evidence_video_ids"]["items"]["enum"] = allowed_ids
    if require_topic:
        schema["properties"]["new_topics"]["minItems"] = 1
    return schema


def parse_model_output(output: str) -> Any:
    """Aceita JSON puro ou delimitado por crases, preservando a resposta original no log."""
    content = output.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content).strip()
    if not content:
        raise ValueError("Resposta final vazia: o servidor não retornou o JSON dos tópicos")
    return json.loads(content)


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().casefold())


def load_initial_taxonomy(path: Path | None) -> list[dict]:
    """Importa topics_final.json, preservando os tópicos e suas evidências históricas."""
    if path is None:
        return []
    try:
        data = read_json(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido na taxonomia inicial {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("topics"), list):
        raise ValueError(f"Taxonomia inicial {path} deve ser um objeto com a lista topics de topics_final.json")
    topics = data["topics"]
    if "total_topics" in data and (type(data["total_topics"]) is not int or data["total_topics"] != len(topics)):
        raise ValueError(f"Taxonomia inicial {path}: total_topics difere do tamanho de topics")
    used_ids: set[str] = set()
    used_labels: set[str] = set()
    required = {"topic_id", "label", "description", "evidence_video_ids", "created_in_batch"}
    for index, topic in enumerate(topics, 1):
        prefix = f"Taxonomia inicial {path}, tópico {index}"
        if not isinstance(topic, dict) or not required.issubset(topic):
            raise ValueError(f"{prefix}: precisa conter os campos {sorted(required)}")
        topic_id = topic["topic_id"]
        if not isinstance(topic_id, str) or not re.fullmatch(r"T[0-9]+", topic_id) or int(topic_id[1:]) < 1:
            raise ValueError(f"{prefix}: topic_id deve seguir o formato T001, T002, ...")
        if topic_id in used_ids:
            raise ValueError(f"{prefix}: topic_id duplicado: {topic_id}")
        for key in ("label", "description"):
            if not isinstance(topic[key], str) or not topic[key].strip():
                raise ValueError(f"{prefix}: {key} deve ser texto não vazio")
        label = normalize_label(topic["label"])
        if label in used_labels:
            raise ValueError(f"{prefix}: rótulo duplicado: {topic['label']}")
        evidence = topic["evidence_video_ids"]
        if (not isinstance(evidence, list) or not evidence
                or not all(isinstance(item, str) and item.strip() and not re.search(r"\s", item) for item in evidence)):
            raise ValueError(f"{prefix}: evidence_video_ids deve conter IDs não vazios")
        if len(evidence) != len(set(evidence)):
            raise ValueError(f"{prefix}: evidence_video_ids contém IDs repetidos")
        if type(topic["created_in_batch"]) is not int or topic["created_in_batch"] < 1:
            raise ValueError(f"{prefix}: created_in_batch deve ser um inteiro positivo")
        used_ids.add(topic_id)
        used_labels.add(label)
    return topics


def validate_result(result: Any, batch: list[dict], existing: list[dict]) -> None:
    if not isinstance(result, dict) or set(result) != {"new_topics"} or not isinstance(result["new_topics"], list):
        raise ValueError("A resposta deve ser um objeto com a lista new_topics")
    if not existing and not result["new_topics"]:
        raise ValueError("A taxonomia está vazia: gere pelo menos um tópico sustentado pelos títulos do lote")
    allowed_ids = {r["video_id"] for r in batch}
    used_labels = {normalize_label(t["label"]) for t in existing}
    for i, topic in enumerate(result["new_topics"], 1):
        if not isinstance(topic, dict) or set(topic) != {"label", "description", "evidence_video_ids"}:
            raise ValueError(f"Tópico {i} não tem o esquema correto")
        label, description, evidence = topic["label"], topic["description"], topic["evidence_video_ids"]
        if not isinstance(label, str) or not label.strip() or not isinstance(description, str) or not description.strip():
            raise ValueError(f"Tópico {i} sem label/description válido")
        if not isinstance(evidence, list) or not evidence or not all(isinstance(x, str) for x in evidence):
            raise ValueError(f"Evidências inválidas para {label}")
        if len(evidence) != len(set(evidence)) or any(x not in allowed_ids for x in evidence):
            raise ValueError(f"Evidências repetidas ou de outro lote: {label}")
        key = normalize_label(label)
        if key in used_labels:
            raise ValueError(f"Tópico duplicado: {label}")
        used_labels.add(key)


def response_usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        raise ValueError("Resposta sem dados de uso; não é possível consolidar métricas")
    return {
        "prompt_tokens": int(usage.prompt_tokens or 0),
        "completion_tokens": int(usage.completion_tokens or 0),
        "total_tokens": int(usage.total_tokens or 0),
        # Ausência de detalhamento significa desconhecido, não zero.
        "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
        "cached_tokens": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None),
    }


def metrics(usage: dict[str, int | None], elapsed: float) -> dict[str, Any]:
    return {
        **usage,
        "elapsed_seconds": round(elapsed, 6),
        "completion_tokens_per_second_e2e": round(usage["completion_tokens"] / elapsed, 4) if elapsed > 0 else 0.0,
    }


def write_attempt_log(
    log_dir: Path, batch_index: int, attempt: int, prompt: str,
    output: str, status: str, usage: dict | None, elapsed: float,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = log_dir / f"batch_{batch_index:03d}_attempt_{attempt:02d}_{stamp}.txt"
    content = (
        f"Horário UTC: {now_utc()}\nLote: {batch_index}\nTentativa: {attempt}\n"
        f"Status: {status}\nTempo: {elapsed:.3f}s\n"
        f"Uso: {json.dumps(usage, ensure_ascii=False)}\n"
        f"\n=== SYSTEM ===\n{SYSTEM_PROMPT}\n"
        f"\n=== USER ===\n{prompt}\n"
        f"\n=== OUTPUT ===\n{output}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def generation_parameters(cfg: argparse.Namespace) -> dict[str, Any]:
    """Parâmetros efetivamente enviados; temperatura explícita somente no vLLM."""
    parameters = {
        "reasoning_effort": cfg.reasoning_effort,
        "max_completion_tokens": cfg.max_tokens,
    }
    if cfg.provider == "vllm":
        parameters["temperature"] = cfg.temperature
    return parameters


def completion_request(cfg: argparse.Namespace, prompt: str, batch: list[dict],
                       require_topic: bool = False) -> dict[str, Any]:
    return {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        **generation_parameters(cfg),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "topic_generation", "strict": True,
                "schema": schema_for_batch(batch, require_topic=require_topic),
            },
        },
    }


def protocol_config(cfg: argparse.Namespace) -> dict[str, Any]:
    """Identifica o procedimento, independentemente do modelo e do provedor."""
    return {
        "prompt_version": PROMPT_VERSION,
        "implementation_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "system_prompt": SYSTEM_PROMPT,
        "output_schema": OUTPUT_SCHEMA,
        "api": "chat.completions",
        "structured_output": "response_format.json_schema.strict",
        "schema_evidence_id_constraint": "enum_per_batch",
        "duplicate_handling": "reject_and_retry",
        "retry_policy": "original_prompt_plus_last_validation_error",
        "json_parsing": "whole_json_with_optional_markdown_fence",
        "empty_taxonomy_policy": "require_at_least_one_topic",
        "batch_size": cfg.batch_size,
        "max_attempts": cfg.max_attempts,
        "shuffle_seed": cfg.shuffle_seed,
        "generation_parameters": generation_parameters(cfg),
        "sampling_policy": "explicit_temperature_for_vllm; provider_default_for_openai; top_p/seed not sent",
        "request_timeout": cfg.request_timeout,
        "sdk_max_retries": cfg.sdk_max_retries,
        "checkpoint_enabled": False,
        "output_policy": "overwrite_selected_experiments",
        "initial_taxonomy_policy": "per_experiment; preserve_topics; process_all_titles",
    }


def call_with_retries(
    client: Any, cfg: argparse.Namespace, existing: list[dict], batch: list[dict],
    batch_index: int, log_dir: Path,
) -> tuple[dict, Any, list[dict], dict[str, Any]]:
    original_prompt = build_prompt(existing, batch)
    previous_error = None
    failures: list[dict] = []
    for attempt in range(1, cfg.max_attempts + 1):
        prompt = original_prompt
        if previous_error:
            prompt += (
                "\n\nATENÇÃO: a saída anterior foi rejeitada pelo validador: "
                f"{previous_error}. Refaça a resposta como JSON válido sem duplicatas e com evidências do lote atual."
            )
        started = time.monotonic()
        response = None
        output = ""
        usage = None
        try:
            response = client.chat.completions.create(**completion_request(cfg, prompt, batch, require_topic=not existing))
            if getattr(response, "usage", None) is not None:
                usage = response_usage(response)
            if not response.choices:
                raise ValueError("Servidor não retornou choices")
            choice = response.choices[0]
            output = choice.message.content or ""
            if not isinstance(output, str):
                raise ValueError("Conteúdo da resposta não é texto JSON")
            if usage is None:
                raise ValueError("Resposta sem dados de uso; não é possível consolidar métricas")
            if choice.finish_reason != "stop":
                raise ValueError(
                    f"Geração não concluída: finish_reason={choice.finish_reason!r}. "
                    "Se houve truncamento, aumente max_completion_tokens no arquivo compartilhado "
                    "e verifique o limite de contexto do servidor. O limite inclui raciocínio."
                )
            if getattr(choice.message, "refusal", None):
                raise ValueError("O modelo recusou a geração")
            result = parse_model_output(output)
            validate_result(result, batch, existing)
            elapsed = time.monotonic() - started
            write_attempt_log(log_dir, batch_index, attempt, prompt, output, "OK", usage, elapsed)
            return result, response, failures, metrics(usage, elapsed)
        except ValueError as exc:
            elapsed = time.monotonic() - started
            previous_error = str(exc)
            log = write_attempt_log(log_dir, batch_index, attempt, prompt, output, f"INVALID: {exc}", usage, elapsed)
            failures.append({
                "attempt": attempt, "error": str(exc), "usage": usage,
                "elapsed_seconds": round(elapsed, 6), "log_file": str(log),
            })
            print(f"[lote {batch_index}] tentativa {attempt} inválida: {exc}", flush=True)
            print(f"[lote {batch_index}] início da resposta: {output[:160]!r} | log: {log}", flush=True)
        except Exception as exc:
            elapsed = time.monotonic() - started
            write_attempt_log(log_dir, batch_index, attempt, prompt, output,
                              f"API_ERROR {type(exc).__name__}: {exc}", usage, elapsed)
            raise
    raise RuntimeError(f"Lote {batch_index} falhou em {cfg.max_attempts} tentativas: {previous_error}")


CSV_FIELDS = [
    "batch", "titles_in_batch", "new_topics", "response_id", "model", "prompt_tokens",
    "completion_tokens", "total_tokens", "reasoning_tokens", "cached_tokens",
    "elapsed_seconds", "completion_tokens_per_second_e2e", "failed_attempts",
    "failed_prompt_tokens", "failed_completion_tokens", "failed_total_tokens", "failed_elapsed_seconds",
]


def export_report(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(client: Any, cfg: argparse.Namespace, records: list[dict], name: str,
                   order: str, seed: int | None, initial_topics: list[dict] | None = None,
                   initial_taxonomy_file: Path | None = None) -> dict:
    """Taxonomia independente, opcionalmente inicializada com tópicos importados."""
    topics = copy.deepcopy(initial_topics) if initial_topics is not None else []
    initial_count = len(topics)
    next_topic_number = max((int(topic["topic_id"][1:]) for topic in topics), default=0) + 1
    initial_info = {
        "initial_taxonomy_file": str(initial_taxonomy_file) if initial_taxonomy_file is not None else None,
        "initial_taxonomy_hash": sha256_json(topics),
        "initial_topics": initial_count,
    }
    directory = cfg.output / name
    log_dir = directory / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    batches = [records[i:i + cfg.batch_size] for i in range(0, len(records), cfg.batch_size)]
    input_hash = sha256_json(records)
    protocol = protocol_config(cfg)
    protocol_hash = sha256_json(protocol)
    comparison_hash = sha256_json({
        "input_hash": input_hash, "protocol_hash": protocol_hash, "experiment": name,
        "initial_taxonomy_hash": initial_info["initial_taxonomy_hash"],
    })
    experiment_info = {
        "experiment": name, "order_description": order, "shuffle_seed": seed,
        "input_hash": input_hash, "total_titles": len(records),
        **initial_info,
    }
    model_source = os.getenv("MODEL_ID", cfg.model) if cfg.provider == "vllm" else cfg.model
    config = {
        "provider": cfg.provider, "model": cfg.model, "model_source": model_source,
        "base_url": cfg.base_url, "protocol": protocol, "protocol_hash": protocol_hash,
        "comparison_hash": comparison_hash, "experiment": experiment_info,
    }
    if cfg.provider == "vllm":
        config["server_environment"] = {
            key: os.environ[key] for key in (
                "MODEL_ID", "REASONING_PARSER", "GENERATION_CONFIG", "MAX_MODEL_LEN", "DTYPE",
                "TENSOR_PARALLEL_SIZE", "TOKENIZER_MODE", "CONFIG_FORMAT", "LOAD_FORMAT", "ATTENTION_BACKEND",
            ) if key in os.environ
        }
    config_hash = sha256_json(config)
    atomic_json(directory / "order_manifest.json", {
        **experiment_info, "video_ids_in_order": [r["video_id"] for r in records],
    })
    atomic_json(log_dir / "00_config.json", config)
    atomic_json(directory / "initial_taxonomy.json", {**initial_info, "topics": topics})

    report_rows: list[dict] = []
    batch_dir = directory / "batches"
    print(f"\n=== {name}: {len(records)} títulos, {len(batches)} lotes, {initial_count} tópicos iniciais ===", flush=True)
    with tqdm(total=len(batches), desc=name, unit="lote") as progress:
        for i, batch in enumerate(batches, 1):
            result, response, failures, measurement = call_with_retries(
                client, cfg, topics, batch, i, log_dir,
            )
            for topic in result["new_topics"]:
                topics.append({
                    "topic_id": f"T{next_topic_number:03d}",
                    "label": topic["label"].strip(), "description": topic["description"].strip(),
                    "evidence_video_ids": topic["evidence_video_ids"], "created_in_batch": i,
                })
                next_topic_number += 1
            # Arquivo de auditoria; nunca é carregado para retomar uma execução.
            atomic_json(batch_dir / f"batch_{i:03d}.json", {
                "batch_index": i, "response_id": response.id, "model": response.model,
                "titles_in_batch": len(batch), "metrics": measurement,
                "failed_attempts": failures, "result": result,
            })
            report_rows.append({
                "batch": i, "titles_in_batch": len(batch), "new_topics": len(result["new_topics"]),
                "response_id": response.id, "model": response.model, **measurement,
                "failed_attempts": len(failures),
                **{
                    f"failed_{key}": sum(int((f.get("usage") or {}).get(key, 0) or 0) for f in failures)
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
                "failed_elapsed_seconds": round(sum(f["elapsed_seconds"] for f in failures), 6),
            })
            export_report(report_rows, directory / "metrics.csv")
            progress.update(1)
            progress.set_postfix(topics=len(topics), tokens=measurement["total_tokens"])

    final = {
        "provider": cfg.provider, "model": cfg.model, "model_source": model_source,
        "prompt_version": PROMPT_VERSION, "reasoning_effort": cfg.reasoning_effort,
        "generation_parameters": generation_parameters(cfg),
        "experiment": experiment_info, "input_file": str(cfg.input),
        "input_hash": input_hash, "config_hash": config_hash,
        "protocol_hash": protocol_hash, "comparison_hash": comparison_hash,
        "total_titles": len(records), "batch_size": cfg.batch_size,
        "total_batches": len(batches), "total_topics": len(topics), "topics": topics,
        **initial_info, "generated_topics": len(topics) - initial_count,
    }
    atomic_json(directory / "topics_final.json", final)
    summary = {
        "experiment": name, "provider": cfg.provider, "model": cfg.model, "model_source": model_source,
        "generation_parameters": generation_parameters(cfg),
        "input_hash": input_hash, "protocol_hash": protocol_hash, "comparison_hash": comparison_hash,
        "total_titles": len(records), "total_batches": len(batches), "total_topics": len(topics),
        **initial_info, "generated_topics": len(topics) - initial_count,
        "total_failed_attempts": sum(r["failed_attempts"] for r in report_rows),
        "checkpoint_enabled": False,
    }
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        successful = sum(r[key] for r in report_rows)
        failed = sum(r[f"failed_{key}"] for r in report_rows)
        total_key = "total_tokens" if key == "total_tokens" else f"total_{key}"
        summary.update({f"successful_{key}": successful, f"failed_{key}": failed, total_key: successful + failed})
    successful_seconds = sum(r["elapsed_seconds"] for r in report_rows)
    failed_seconds = sum(r["failed_elapsed_seconds"] for r in report_rows)
    summary.update({
        "successful_elapsed_seconds": round(successful_seconds, 6),
        "failed_elapsed_seconds": round(failed_seconds, 6),
        "total_elapsed_seconds_e2e": round(successful_seconds + failed_seconds, 6),
    })
    atomic_json(directory / "summary.json", summary)
    print("Resumo:", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def wait_for_vllm(client: Any, model: str, timeout_seconds: float, interval: float = 5.0) -> None:
    started = time.monotonic()
    last_error = "não respondeu"
    print(f"Aguardando vLLM disponibilizar o modelo {model!r}...", flush=True)
    while True:
        try:
            available = [m.id for m in client.models.list().data]
            if model in available:
                print(f"vLLM pronto: {model}", flush=True)
                return
            last_error = f"API pronta, mas modelos disponíveis são {available}; solicitado: {model!r}"
        except Exception as exc:
            last_error = str(exc)
        if time.monotonic() - started >= timeout_seconds:
            raise RuntimeError(f"vLLM indisponível após {timeout_seconds}s: {last_error}")
        time.sleep(interval)


def load_experiment_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    required = {
        "batch_size", "max_attempts", "shuffle_seed", "max_completion_tokens",
        "reasoning_effort", "request_timeout", "sdk_max_retries",
    }
    optional = {"vllm_temperature"}
    if not isinstance(config, dict) or not required.issubset(config) or set(config) - required - optional:
        raise ValueError(f"{path} deve conter os campos {sorted(required)} e pode incluir {sorted(optional)}")
    config.setdefault("vllm_temperature", 0.0)
    for key in ("batch_size", "max_attempts", "max_completion_tokens", "sdk_max_retries", "shuffle_seed"):
        if type(config[key]) is not int:
            raise ValueError(f"{key} precisa ser um inteiro")
    if min(config[key] for key in ("batch_size", "max_attempts", "max_completion_tokens")) < 1:
        raise ValueError("batch_size, max_attempts e max_completion_tokens devem ser positivos")
    if config["sdk_max_retries"] < 0:
        raise ValueError("sdk_max_retries não pode ser negativo")
    temperature = config["vllm_temperature"]
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
        raise ValueError("vllm_temperature deve ser um número finito entre 0 e 2")
    if config["reasoning_effort"] not in ("low", "medium", "high", "xhigh", "max"):
        raise ValueError("reasoning_effort deve ser low, medium, high, xhigh ou max")
    timeout = config["request_timeout"]
    if type(timeout) not in (int, float) or not 0 < timeout < float("inf"):
        raise ValueError("request_timeout deve ser positivo e finito")
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=Path(os.getenv("EXPERIMENT_CONFIG", str(DEFAULT_CONFIG))))
    bootstrap.add_argument("--provider", choices=["openai", "vllm"], default="vllm")
    preliminary, _ = bootstrap.parse_known_args(argv)
    defaults = load_experiment_config(preliminary.config)
    is_openai = preliminary.provider == "openai"
    parser = argparse.ArgumentParser(
        parents=[bootstrap], description="Mesmo experimento de tópicos para OpenAI e vLLM, com raciocínio",
    )
    parser.add_argument("--input", type=Path, default=Path(os.getenv("INPUT_FILE", "data/geracao.jsonl")))
    parser.add_argument("--output", type=Path, default=Path(os.getenv("OUTPUT_DIR", f"outputs/{preliminary.provider}")))
    parser.add_argument("--model", default="gpt-6.1-sol" if is_openai else os.getenv("SERVED_MODEL_NAME", "topic-local"))
    # A execução OpenAI não herda o endpoint/chave do contêiner local.
    parser.add_argument("--base-url", default="https://api.openai.com/v1" if is_openai else os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY") if is_openai else os.getenv("VLLM_API_KEY", "local-token"))
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--max-attempts", type=int, default=defaults["max_attempts"])
    parser.add_argument("--shuffle-seed", type=int, default=defaults["shuffle_seed"])
    parser.add_argument("--max-tokens", "--max-completion-tokens", dest="max_tokens", type=int, default=defaults["max_completion_tokens"])
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default=defaults["reasoning_effort"])
    parser.add_argument(
        "--temperature", type=float, default=None if is_openai else defaults["vllm_temperature"],
        help="temperatura do vLLM; padrão definido em vllm_temperature no JSON de configuração",
    )
    parser.add_argument("--request-timeout", type=float, default=defaults["request_timeout"])
    parser.add_argument("--sdk-max-retries", type=int, default=defaults["sdk_max_retries"])
    parser.add_argument("--startup-timeout", type=float, default=float(os.getenv("STARTUP_TIMEOUT", "900")))
    parser.add_argument("--experiment", choices=["both", "original", "shuffled"], default=os.getenv("EXPERIMENT", "both"))
    parser.add_argument(
        "--initial-taxonomy-original", type=Path, default=os.getenv("INITIAL_TAXONOMY_ORIGINAL") or None,
        help="topics_final.json com os tópicos iniciais do experimento em ordem original",
    )
    parser.add_argument(
        "--initial-taxonomy-shuffled", type=Path, default=os.getenv("INITIAL_TAXONOMY_SHUFFLED") or None,
        help="topics_final.json com os tópicos iniciais do experimento em ordem embaralhada",
    )
    cfg = parser.parse_args(argv)
    if cfg.temperature is not None:
        if cfg.provider != "vllm":
            parser.error("--temperature está disponível somente para o provedor vllm")
        if not 0 <= cfg.temperature <= 2:
            parser.error("temperature deve ser um número finito entre 0 e 2")
    if min(cfg.batch_size, cfg.max_attempts, cfg.max_tokens) < 1 or cfg.sdk_max_retries < 0:
        parser.error("batch-size, max-attempts e max-tokens devem ser positivos; sdk-max-retries deve ser não negativo")
    if not 0 < cfg.request_timeout < float("inf") or not 0 < cfg.startup_timeout < float("inf"):
        parser.error("timeouts devem ser positivos e finitos")
    return cfg


def create_client(cfg: argparse.Namespace) -> Any:
    if not cfg.api_key:
        raise ValueError("Defina OPENAI_API_KEY para executar o provedor OpenAI")
    from openai import OpenAI

    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key,
                  timeout=cfg.request_timeout, max_retries=cfg.sdk_max_retries)


def validate_backend_config(cfg: argparse.Namespace) -> None:
    """Falha antes do processamento se o tokenizer Mistral rejeitaria o esforço."""
    if cfg.provider != "vllm":
        return
    model_source = os.getenv("MODEL_ID", cfg.model).casefold()
    is_mistral = (
        os.getenv("REASONING_PARSER", "").casefold() == "mistral"
        or os.getenv("TOKENIZER_MODE", "").casefold() == "mistral"
        or model_source.startswith("mistralai/")
    )
    if is_mistral and cfg.reasoning_effort != "high":
        raise ValueError(
            "O tokenizer Mistral aceita apenas reasoning_effort='none' ou 'high'. "
            "Como este experimento usa raciocínio nos dois modelos, defina "
            "reasoning_effort='high' em experiment_config.json para ambos."
        )


def clear_previous_results(cfg: argparse.Namespace) -> None:
    """Limpa somente as saídas gerenciadas dos experimentos selecionados."""
    cfg.output.mkdir(parents=True, exist_ok=True)
    directories = []
    if cfg.experiment in ("both", "original"):
        directories.append(cfg.output / "experiment_1_original_order")
    if cfg.experiment in ("both", "shuffled"):
        directories.append(cfg.output / "experiment_2_shuffled_order")
    files = [cfg.output / "protocol.json", cfg.output / "summary_all.json"]
    # Valida todos os destinos antes de apagar para evitar limpeza parcial.
    for path in directories + files:
        if path.is_symlink():
            raise RuntimeError(f"Saída gerenciada é link simbólico: {path}")
        if path.exists():
            expected = path.is_dir() if path in directories else path.is_file()
            if not expected:
                raise RuntimeError(f"Saída gerenciada tem tipo inesperado: {path}")
    for path in directories:
        if path.exists():
            shutil.rmtree(path)
            print(f"[gerador] Saída anterior removida: {path}", flush=True)
    for path in files:
        path.unlink(missing_ok=True)


def run(cfg: argparse.Namespace, client: Any = None) -> dict[str, Any]:
    """Entrada comum do notebook e da CLI; sobrescreve as saídas a cada execução."""
    validate_backend_config(cfg)
    records = load_titles(cfg.input)
    # Carrega todas as taxonomias selecionadas antes de limpar saídas ou chamar a API.
    # A origem pode ser um topics_final.json que será sobrescrito nesta execução.
    initial_taxonomies = {
        order: load_initial_taxonomy(getattr(cfg, f"initial_taxonomy_{order}", None))
        for order in ("original", "shuffled") if cfg.experiment in ("both", order)
    }
    if client is None:
        client = create_client(cfg)
    clear_previous_results(cfg)
    if cfg.provider == "vllm":
        wait_for_vllm(client, cfg.model, cfg.startup_timeout)
    run_dir = cfg.output
    protocol = protocol_config(cfg)
    protocol_hash = sha256_json(protocol)
    print(f"[gerador] {cfg.provider} | {cfg.model} | reasoning={cfg.reasoning_effort}", flush=True)
    if cfg.provider == "vllm":
        print(f"[gerador] temperature={cfg.temperature}", flush=True)
    print(f"[gerador] Protocolo: {protocol_hash}\n[gerador] Resultados: {run_dir}", flush=True)
    atomic_json(run_dir / "protocol.json", protocol)
    summaries = []
    if cfg.experiment in ("both", "original"):
        summaries.append(run_experiment(client, cfg, records,
                                       "experiment_1_original_order", "ordem original", None,
                                       initial_taxonomies["original"], cfg.initial_taxonomy_original))
    if cfg.experiment in ("both", "shuffled"):
        shuffled = list(records)
        random.Random(cfg.shuffle_seed).shuffle(shuffled)
        if len(records) > 1 and shuffled == records:
            raise RuntimeError("O embaralhamento retornou a ordem original; altere shuffle_seed nos dois provedores")
        summaries.append(run_experiment(client, cfg, shuffled,
                                       "experiment_2_shuffled_order", "ordem embaralhada", cfg.shuffle_seed,
                                       initial_taxonomies["shuffled"], cfg.initial_taxonomy_shuffled))
    report = {
        "run_directory": str(run_dir), "provider": cfg.provider, "model": cfg.model,
        "input_hash": sha256_json(records), "protocol_hash": protocol_hash,
        "experiments": summaries, "generated_at": now_utc(),
    }
    atomic_json(run_dir / "summary_all.json", report)
    print("\nEXPERIMENTO(S) CONCLUÍDO(S). Resultados em:", run_dir, flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
