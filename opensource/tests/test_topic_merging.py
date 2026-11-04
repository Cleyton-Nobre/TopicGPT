"""Verifica a mesclagem de ponta a ponta sem servidor, downloads ou GPU."""
import ast
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np

import topic_merging as merging


class Encoder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.texts = []

    def encode(self, texts, **kwargs):
        self.texts = texts
        return np.asarray(self.vectors, dtype=float)


def response(result=None, *, content=None, finish_reason="stop", refusal=None):
    if content is None and result is not None:
        content = json.dumps(result, ensure_ascii=False)
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, refusal=refusal, reasoning="Raciocínio local."),
    )])


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="topic-local")]))

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


class TopicMergingTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.topics = [
            {"topic_id": "T001", "label": "Aplicação", "description": "Como aplicar o medicamento."},
            {"topic_id": "T002", "label": "Doses", "description": "Como ajustar a dose."},
            {"topic_id": 3, "label": "Preço", "description": "Custo do tratamento."},
        ]
        self.result = {"groups": [
            {"topic_ids": ["T001", "T002"], "label": "Uso do medicamento",
             "description": "Aplicação e ajuste de doses.", "justification": "Aspectos práticos do uso."},
            {"topic_ids": [3], "label": "Preço", "description": "Custo do tratamento.",
             "justification": "Tema financeiro distinto."},
        ]}
        self.input = self.root / "input.json"
        self.input.write_text(json.dumps({"topics": self.topics}), encoding="utf-8")
        self.output = self.root / "output" / "merged.json"
        self.encoder = Encoder([[3, 0], [4, 3], [0, 5]])
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for redirect in (redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            redirect.__enter__()
            self.addCleanup(redirect.__exit__, None, None, None)

    def cfg(self, *extra):
        return merging.parse_args(["--input", str(self.input), "--output", str(self.output), *extra])

    def test_prompts_and_embedding_model_match_reference_notebook(self):
        notebook_path = Path(merging.__file__).resolve().parents[1] / "merging.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        constants = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                try:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        self.assertEqual(merging.MERGE_SYSTEM_PROMPT, constants["MERGE_SYSTEM_PROMPT"])
        self.assertEqual(merging.CONTEUDO_DOMINIO, constants["CONTEUDO_DOMINIO"])
        self.assertEqual(self.cfg().similarity_threshold, constants["LIMIAR_SIMILARIDADE"])
        self.assertIn(merging.DEFAULT_EMBEDDING_MODEL, source)

    def test_pairs_use_cosine_inclusive_threshold_and_descending_order(self):
        pairs = merging.find_similar_pairs(self.topics, 0.6, "test", encoder=self.encoder)
        self.assertEqual(len(pairs), 2)
        self.assertEqual([(p["topico_1"]["topic_id"], p["topico_2"]["topic_id"]) for p in pairs],
                         [("T001", "T002"), ("T002", 3)])
        np.testing.assert_allclose([pair["score"] for pair in pairs], [0.8, 0.6])
        self.assertEqual(self.encoder.texts, [f"{t['label']}: {t['description']}" for t in self.topics])
        self.assertEqual(merging.find_similar_pairs(self.topics[:1], 0.5, "test"), [])

    def test_invalid_embeddings_are_rejected(self):
        for vectors in ([[0, 0]] * 3, [[1, float("nan")]] * 3, [[1, 0]], [1, 2, 3]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                merging.find_similar_pairs(self.topics, 0.5, "test", encoder=Encoder(vectors))

    def test_end_to_end_retries_invalid_allocation_and_exports_only_valid_groups(self):
        invalid = copy.deepcopy(self.result)
        invalid["groups"].pop()
        client = Client([response(invalid), response(self.result)])
        cfg = self.cfg()
        original = self.input.read_bytes()
        result = merging.run(cfg, client=client, encoder=self.encoder)
        self.assertEqual(result, self.result)
        self.assertEqual(json.loads(cfg.output.read_text()), self.result)
        self.assertEqual(self.input.read_bytes(), original)
        self.assertEqual(len(client.requests), 2)
        first, retry = client.requests
        self.assertNotIn("ATENÇÃO:", first["messages"][1]["content"])
        self.assertIn("não alocados: ['3']", retry["messages"][1]["content"])
        self.assertEqual(first["reasoning_effort"], "high")
        self.assertEqual(first["max_completion_tokens"], 4096)
        self.assertEqual(first["model"], "topic-local")
        self.assertEqual(first["temperature"], 0.0)
        schema = first["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertEqual(schema["schema"]["properties"]["groups"]["items"]["properties"]["topic_ids"]["items"]["enum"],
                         ["T001", "T002", 3])
        saved_prompt = cfg.prompt_file.read_text()
        self.assertIn(first["messages"][0]["content"].strip(), saved_prompt)
        self.assertIn(first["messages"][1]["content"].strip(), saved_prompt)
        self.assertIn('"groups"', saved_prompt)
        logs = json.loads(cfg.log_file.read_text())
        self.assertEqual(len(logs), 2)
        self.assertIn("error", logs[0])
        self.assertNotIn("error", logs[1])
        self.assertEqual(logs[1]["reasoning"], "Raciocínio local.")

    def test_no_candidates_still_sends_all_topics_for_global_consolidation(self):
        client = Client([response(self.result)])
        merging.run(self.cfg("--similarity-threshold", "1"), client=client, encoder=self.encoder)
        prompt = client.requests[0]["messages"][1]["content"]
        for topic in self.topics:
            self.assertIn(f"[{topic['topic_id']}] {topic['label']}", prompt)
        self.assertNotIn("(Similaridade:", prompt)

    def test_invalid_ids_and_incomplete_groups_are_rejected(self):
        for ids in (["T001", "T001", "T002"], ["T001", "T002", "inventado"],
                    ["T001"], [], ["T001", "T002", True], ["T001", "T002", 3.0]):
            invalid = copy.deepcopy(self.result)
            invalid["groups"][0]["topic_ids"] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                merging.validate_result(invalid, self.topics)
        invalid = copy.deepcopy(self.result)
        invalid["groups"][1]["topic_ids"] = ["3"]
        with self.assertRaisesRegex(ValueError, "tipo original"):
            merging.validate_result(invalid, self.topics)
        invalid = copy.deepcopy(self.result)
        invalid["groups"][1]["topic_ids"] = [3, "T001"]
        with self.assertRaisesRegex(ValueError, "Duplicados"):
            merging.validate_result(invalid, self.topics)
        for field in ("label", "description", "justification"):
            invalid = copy.deepcopy(self.result)
            invalid["groups"][0][field] = " "
            with self.subTest(field=field), self.assertRaises(ValueError):
                merging.validate_result(invalid, self.topics)
        for invalid in ([], {"groups": []}, {"groups": {}}, {"groups": [{}]},
                        {"groups": [None]}, {**self.result, "extra": 1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                merging.validate_result(invalid, self.topics)

    def test_failed_attempts_preserve_previous_output_and_log_raw_response(self):
        cfg = self.cfg("--max-attempts", "2")
        cfg.output.parent.mkdir()
        cfg.output.write_text("saída anterior", encoding="utf-8")
        client = Client([response(content="{JSON incompleto"), response(self.result, finish_reason="length")])
        with self.assertRaisesRegex(RuntimeError, "após 2 tentativas"):
            merging.run(cfg, client=client, encoder=self.encoder)
        self.assertEqual(cfg.output.read_text(), "saída anterior")
        logs = json.loads(cfg.log_file.read_text())
        self.assertEqual(logs[0]["content"], "{JSON incompleto")
        self.assertEqual(logs[1]["finish_reason"], "length")
        self.assertTrue(all("error" in log for log in logs))

    def test_empty_refused_or_missing_choices_responses_are_retried(self):
        bad_responses = [response(content=""), response(refusal="Recusado"), SimpleNamespace(choices=[])]
        for bad_response in bad_responses:
            client = Client([bad_response, response(self.result)])
            merging.run(self.cfg(), client=client, encoder=self.encoder)
            self.assertEqual(len(client.requests), 2)

    def test_transport_failure_is_logged_and_propagated(self):
        cfg = self.cfg()
        client = Client([RuntimeError("servidor indisponível")])
        with self.assertRaisesRegex(RuntimeError, "servidor indisponível"):
            merging.run(cfg, client=client, encoder=self.encoder)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(json.loads(cfg.log_file.read_text())[0]["error"], "servidor indisponível")
        self.assertFalse(cfg.output.exists())

    def test_markdown_json_is_accepted_and_reasoning_can_be_omitted(self):
        client = Client([response(content="```json\n" + json.dumps(self.result) + "\n```")])
        merging.run(self.cfg("--reasoning-effort", "omit"), client=client, encoder=self.encoder)
        self.assertNotIn("reasoning_effort", client.requests[0])

    def test_invalid_input_fails_before_embeddings_or_api_or_file_writes(self):
        invalid_inputs = [[], {}, {"topics": []}, {"topics": [None]}, {"topics": [{}]},
                          {"topics": [self.topics[0]] * 2}]
        for field, value in (("topic_id", True), ("topic_id", 1.5), ("topic_id", " "),
                             ("label", ""), ("description", None)):
            invalid_inputs.append({"topics": [{**self.topics[0], field: value}]})
        with patch.object(merging, "find_similar_pairs") as embeddings, patch.object(merging, "wait_for_vllm") as wait:
            for data in invalid_inputs:
                self.input.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(data=data), self.assertRaises(ValueError):
                    merging.run(self.cfg())
            embeddings.assert_not_called()
            wait.assert_not_called()
        self.assertFalse(self.output.parent.exists())

    def test_cli_rejects_invalid_parameters_and_overlapping_files(self):
        for args in (("--similarity-threshold", "nan"), ("--similarity-threshold", "1.1"),
                     ("--max-tokens", "0"), ("--max-attempts", "0"), ("--sdk-max-retries", "-1"),
                     ("--temperature", "inf"), ("--startup-timeout", "0"), ("--request-timeout", "nan"),
                     ("--output", str(self.input)), ("--prompt-file", str(self.output)),
                     ("--log-file", str(self.input))):
            with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                self.cfg(*args)
            self.assertEqual(error.exception.code, 2)

    def test_local_client_uses_explicit_url_and_key_without_openai_credentials(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unrelated", "OPENAI_BASE_URL": "https://example.invalid"}):
            self.assertEqual(self.cfg().base_url, "http://localhost:8000/v1")
            self.assertEqual(self.cfg().api_key, "local-token")
        with patch.dict(os.environ, {"VLLM_BASE_URL": "http://custom:8000/v1", "VLLM_API_KEY": "custom-key",
                                     "MERGE_INPUT": str(self.input), "MERGE_OUTPUT": str(self.output)}):
            cfg = merging.parse_args([])
            self.assertEqual(cfg.base_url, "http://custom:8000/v1")
            self.assertEqual(cfg.api_key, "custom-key")
            self.assertEqual(cfg.input, self.input)
            self.assertEqual(cfg.output, self.output)

    def test_docker_entrypoint_selects_merging_without_starting_a_real_server(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        scripts = {
            "chmod": "#!/bin/bash\nexit 0\n",
            "vllm": '''#!/bin/bash
touch "$CAPTURE_DIRECTORY/server-ready"
exec /bin/sleep 30
''',
            "python3": '''#!/bin/bash
printf '%s\\n' "$*" >> "$CAPTURE_DIRECTORY/python-commands.txt"
if [[ "$1" == "-m" ]]; then exit 0; fi
for attempt in {1..50}; do
    if [[ -f "$CAPTURE_DIRECTORY/server-ready" ]]; then exit 0; fi
    sleep 0.02
done
exit 1
''',
        }
        for name, source in scripts.items():
            path = bin_dir / name
            path.write_text(source, encoding="utf-8")
            path.chmod(0o700)
        result = subprocess.run(
            ["/bin/bash", str(Path(merging.__file__).with_name("entrypoint.sh"))],
            env={"PATH": f"{bin_dir}:/usr/bin:/bin", "CAPTURE_DIRECTORY": str(self.root),
                 "TOPIC_TASK": "merging", "MODEL_ID": "mistralai/Ministral-3-14B-Reasoning-2512"},
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "python-commands.txt").read_text().splitlines(), [
            "-m pip install --disable-pip-version-check -r /app/requirements-merging.txt",
            "-u /app/topic_merging.py",
        ])


if __name__ == "__main__":
    unittest.main()
