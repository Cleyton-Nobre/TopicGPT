#!/usr/bin/env python3
"""Consolidação global de merging.ipynb com um modelo aberto servido pelo vLLM."""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Any

from topic_generation import atomic_json, parse_model_output, wait_for_vllm


ROOT = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MERGE_SYSTEM_PROMPT = """
Você é um pesquisador especialista em análise temática.

Você receberá uma lista de tópicos e pares candidatos à mesclagem,
identificados previamente por similaridade semântica.


Sua tarefa é analisar todos os tópicos conjuntamente e produzir
uma consolidação global, seguindo estas regras:

1. Considere todos os pares em conjunto, pois um mesmo tópico
   pode aparecer em diferentes pares. Não tome decisões isoladas.

2. Mescle tópicos que apresentem proximidade temática relevante,
   mesmo que abordem aspectos diferentes de um mesmo assunto.
   Não exija equivalência de significado ou que os tópicos sejam
   praticamente sinônimos. Considere se eles podem ser
   representados por uma categoria temática mais abrangente,
   coerente e informativa.

3. Não presuma que a similaridade é transitiva: tópicos conectados
   por pares semelhantes não precisam pertencer ao mesmo grupo.
   Avalie cada possível grupo conjuntamente, verificando se todos
   os seus integrantes podem ser representados por um conceito
   abrangente e tematicamente coerente.

4. Cada tópico original deve pertencer a exatamente um grupo,
   que pode conter um ou mais tópicos. Não produza sobreposições.

5. Para grupos com mais de um tópico, crie um rótulo conciso
   e uma descrição abrangente que represente os aspectos
   compartilhados e complementares dos tópicos agrupados,
   sem introduzir assuntos que não estejam presentes nos originais.

6. Evite manter tópicos separados quando houver uma possibilidade
   razoável de agrupá-los em uma categoria temática coerente.
   Preserve individualmente apenas os tópicos que apresentem
   um foco suficientemente distinto dos demais.

7. Preserve os IDs originais e apresente uma justificativa breve
   para cada grupo formado ou tópico mantido separado.

8. Retorne exclusivamente um objeto JSON válido, seguindo
   rigorosamente o schema fornecido. Não inclua comentários,
   explicações ou blocos Markdown fora do JSON.
"""

CONTEUDO_DOMINIO = (
    "DOMÍNIO E CONTEXTO DE SAÚDE:\n"
    "Os tópicos abaixo foram extraídos de títulos de vídeos do YouTube em português "
    "sobre medicamentos para emagrecimento (ex.: Ozempic, Mounjaro, "
    "etc.). Leve em consideração as especificidades clínicas, "
    "farmacológicas, comportamentais e sociais deste nicho ao decidir se dois tópicos "
    "possuem equivalência temática central ou apenas compartilham o assunto geral de saúde.\n"
)


def load_topics(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("topics"), list) or not data["topics"]:
        raise ValueError("A entrada deve conter uma lista não vazia de topics, como topics_final.json")
    seen: set[str] = set()
    for index, topic in enumerate(data["topics"], 1):
        if not isinstance(topic, dict):
            raise ValueError(f"Tópico {index} deve ser um objeto")
        topic_id = topic.get("topic_id")
        if type(topic_id) not in (int, str) or not str(topic_id).strip():
            raise ValueError(f"topic_id inválido no tópico {index}")
        if str(topic_id) in seen:
            raise ValueError(f"topic_id duplicado: {topic_id}")
        seen.add(str(topic_id))
        for field in ("label", "description"):
            if not isinstance(topic.get(field), str) or not topic[field].strip():
                raise ValueError(f"{field} deve ser texto não vazio no tópico {topic_id}")
    return data["topics"]


def find_similar_pairs(topics: list[dict], threshold: float, model_name: str,
                       device: str = "cpu", encoder: Any = None) -> list[dict]:
    # A CPU deixa a memória da GPU disponível para o servidor de geração.
    if len(topics) < 2:
        return []
    import numpy as np

    if encoder is None:
        from sentence_transformers import SentenceTransformer

        print(f"Carregando embeddings: {model_name} ({device})...", flush=True)
        encoder = SentenceTransformer(model_name, device=device)
    texts = [f"{topic['label']}: {topic['description']}" for topic in topics]
    embeddings = np.asarray(encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(topics) or not np.isfinite(embeddings).all():
        raise ValueError("Embeddings inválidos para os tópicos informados")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("O modelo retornou embedding com norma zero")
    normalized = embeddings / norms
    similarities = np.clip(normalized @ normalized.T, -1, 1)
    pairs = [
        {"score": float(similarities[i, j]), "topico_1": topics[i], "topico_2": topics[j]}
        for i, j in itertools.combinations(range(len(topics)), 2)
        if similarities[i, j] >= threshold
    ]
    return sorted(pairs, key=lambda pair: pair["score"], reverse=True)


def build_prompt(topics: list[dict], pairs: list[dict]) -> str:
    topics_text = "".join(
        f"[{topic['topic_id']}] {topic['label']}\n    Descrição: {topic['description']}\n"
        for topic in topics
    )
    pairs_text = "".join(
        f"- [{pair['topico_1']['topic_id']}] e [{pair['topico_2']['topic_id']}] "
        f"(Similaridade: {pair['score']:.4f})\n"
        for pair in pairs
    )
    return f"""
{CONTEUDO_DOMINIO}
{topics_text}

Abaixo estão os PARES CANDIDATOS com alta similaridade semântica para sua análise conjunta:
{pairs_text}

Por favor, forneça a consolidação global, garantindo que cada um dos IDs acima seja alocado em exatamente um grupo.
"""


def merge_schema(topics: list[dict]) -> dict:
    """Mesmo formato do notebook, com IDs restritos aos originais pelo schema."""
    fields = {
        "topic_ids": {
            "type": "array", "minItems": 1,
            "items": {"anyOf": [{"type": "integer"}, {"type": "string"}],
                      "enum": [topic["topic_id"] for topic in topics]},
        },
        "label": {"type": "string"},
        "description": {"type": "string"},
        "justification": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": {"groups": {
            "type": "array", "minItems": 1,
            "items": {"type": "object", "properties": fields,
                      "required": list(fields), "additionalProperties": False},
        }},
        "required": ["groups"], "additionalProperties": False,
    }


def validate_result(result: Any, topics: list[dict]) -> None:
    if not isinstance(result, dict) or set(result) != {"groups"}:
        raise ValueError("A resposta deve conter somente o campo groups")
    groups = result["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups deve ser uma lista não vazia")
    received: list[str] = []
    expected = {str(topic["topic_id"]): topic["topic_id"] for topic in topics}
    for index, group in enumerate(groups, 1):
        if not isinstance(group, dict) or set(group) != {"topic_ids", "label", "description", "justification"}:
            raise ValueError(f"Campos inválidos no grupo {index}")
        if not isinstance(group["topic_ids"], list) or not group["topic_ids"]:
            raise ValueError(f"topic_ids deve ser uma lista não vazia no grupo {index}")
        for field in ("label", "description", "justification"):
            if not isinstance(group[field], str) or not group[field].strip():
                raise ValueError(f"{field} deve ser texto não vazio no grupo {index}")
        for topic_id in group["topic_ids"]:
            if type(topic_id) not in (int, str):
                raise ValueError(f"ID inválido no grupo {index}: {topic_id!r}")
            key = str(topic_id)
            if key in expected and type(topic_id) is not type(expected[key]):
                raise ValueError(f"O ID {topic_id!r} deve preservar o tipo original")
            received.append(key)
    counts = Counter(received)
    duplicated = sorted(key for key, count in counts.items() if count > 1)
    unknown = sorted(set(received) - expected.keys())
    missing = sorted(expected.keys() - set(received))
    if duplicated or unknown or missing:
        raise ValueError(
            f"Cada ID deve aparecer exatamente uma vez. Duplicados: {duplicated}; "
            f"inexistentes: {unknown}; não alocados: {missing}"
        )


def consolidate(client: Any, cfg: argparse.Namespace, topics: list[dict], prompt: str, schema: dict) -> dict:
    attempts = []
    error = ""
    for attempt in range(1, cfg.max_attempts + 1):
        user_prompt = prompt
        if error:
            user_prompt += f"\nATENÇÃO: a saída anterior foi rejeitada: {error}\nRefaça a consolidação completa."
        record: dict[str, Any] = {"attempt": attempt, "model": cfg.model, "user_prompt": user_prompt}
        try:
            request = {
                "model": cfg.model,
                "messages": [{"role": "system", "content": MERGE_SYSTEM_PROMPT},
                             {"role": "user", "content": user_prompt}],
                "temperature": cfg.temperature, "max_completion_tokens": cfg.max_tokens,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "topic_merge", "strict": True, "schema": schema,
                }},
            }
            if cfg.reasoning_effort != "omit":
                request["reasoning_effort"] = cfg.reasoning_effort
            response = client.chat.completions.create(**request)
            if not response.choices:
                raise ValueError("O servidor retornou uma resposta sem choices")
            choice = response.choices[0]
            record.update(
                content=choice.message.content,
                reasoning=getattr(choice.message, "reasoning", None) or getattr(choice.message, "reasoning_content", None),
                finish_reason=choice.finish_reason,
                refusal=getattr(choice.message, "refusal", None),
            )
            if record["refusal"]:
                raise ValueError(f"O modelo recusou a solicitação: {record['refusal']}")
            if choice.finish_reason != "stop":
                raise ValueError(
                    f"Geração não concluída: finish_reason={choice.finish_reason!r}. "
                    "Em caso de truncamento, aumente --max-tokens e confira MAX_MODEL_LEN do vLLM."
                )
            if not choice.message.content:
                raise ValueError("O servidor retornou uma resposta final vazia")
            result = parse_model_output(choice.message.content)
            validate_result(result, topics)
        except (ValueError, TypeError) as exc:
            error = str(exc)
            record["error"] = error
            print(f"Tentativa {attempt}/{cfg.max_attempts} rejeitada: {error}", file=sys.stderr)
        except Exception as exc:
            # O SDK já aplica suas retentativas de transporte. Erros de API são propagados.
            record["error"] = str(exc)
            raise
        else:
            return result
        finally:
            attempts.append(record)
            atomic_json(cfg.log_file, attempts)
    raise RuntimeError(f"A consolidação falhou após {cfg.max_attempts} tentativas: {error}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(os.getenv("MERGE_INPUT", str(ROOT / "data/topics_final_order.json"))))
    parser.add_argument("--output", type=Path, default=Path(os.getenv("MERGE_OUTPUT", str(ROOT / "outputs/mesclagem/taxonomia_mesclada.json"))))
    parser.add_argument("--prompt-file", type=Path, help="padrão: prompt_mesclagem.txt ao lado da saída")
    parser.add_argument("--log-file", type=Path, help="padrão: <nome da saída>_attempts.json ao lado da saída")
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME", "topic-local"))
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "local-token"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "omit"], default="high",
                        help="use omit para modelos sem suporte ao parâmetro reasoning_effort")
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--sdk-max-retries", type=int, default=2)
    cfg = parser.parse_args(argv)
    if not -1 <= cfg.similarity_threshold <= 1:
        parser.error("similarity-threshold deve ser finito e estar entre -1 e 1")
    if not 0 <= cfg.temperature <= 2:
        parser.error("temperature deve ser finita e estar entre 0 e 2")
    if min(cfg.max_tokens, cfg.max_attempts) < 1 or cfg.sdk_max_retries < 0:
        parser.error("max-tokens e max-attempts devem ser positivos; sdk-max-retries não pode ser negativo")
    if not all(0 < value < float("inf") for value in (cfg.request_timeout, cfg.startup_timeout)):
        parser.error("timeouts devem ser positivos e finitos")
    cfg.prompt_file = cfg.prompt_file or cfg.output.with_name("prompt_mesclagem.txt")
    cfg.log_file = cfg.log_file or cfg.output.with_name(cfg.output.stem + "_attempts.json")
    paths = [cfg.input, cfg.output, cfg.prompt_file, cfg.log_file]
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("input, output, prompt-file e log-file devem apontar para arquivos distintos")
    return cfg


def run(cfg: argparse.Namespace, client: Any = None, encoder: Any = None) -> dict:
    topics = load_topics(cfg.input)
    pairs = find_similar_pairs(topics, cfg.similarity_threshold, cfg.embedding_model, cfg.embedding_device, encoder)
    print(f"Tópicos originais: {len(topics)}; pares candidatos: {len(pairs)}", flush=True)
    prompt = build_prompt(topics, pairs)
    schema = merge_schema(topics)
    cfg.prompt_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.prompt_file.write_text(
        MERGE_SYSTEM_PROMPT.strip() + "\n\n" + prompt.strip() + "\n\n" +
        json.dumps(schema, ensure_ascii=False, indent=4), encoding="utf-8",
    )
    if client is None:
        from openai import OpenAI

        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key,
                        timeout=cfg.request_timeout, max_retries=cfg.sdk_max_retries)
    wait_for_vllm(client, cfg.model, cfg.startup_timeout)
    result = consolidate(client, cfg, topics, prompt, schema)
    atomic_json(cfg.output, result)
    merged = sum(len(group["topic_ids"]) > 1 for group in result["groups"])
    print(f"Grupos consolidados: {len(result['groups'])}; grupos com mesclagem: {merged}; "
          f"tópicos preservados individualmente: {len(result['groups']) - merged}")
    print(f"Taxonomia salva em: {cfg.output}\nPrompt salvo em: {cfg.prompt_file}\nTentativas salvas em: {cfg.log_file}")
    return result


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    try:
        run(cfg)
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
