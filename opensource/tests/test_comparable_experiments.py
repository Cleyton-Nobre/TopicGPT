"""Verifica o experimento completo com respostas controladas, sem rede ou GPU."""
import copy
import csv
import io
import json
import os
import random
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import topic_generation as experiment


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "topic_generation_two_order_experiments_v4.ipynb"


def response(result, finish_reason="stop", usage=True):
    return SimpleNamespace(
        id="controlled-response", model="controlled-model",
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=json.dumps(result), refusal=None),
        )],
        usage=SimpleNamespace(
            prompt_tokens=20, completion_tokens=10, total_tokens=30,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=4),
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ) if usage else None,
    )


class ControlledClient:
    def __init__(self, responses=None, supported_efforts=None):
        self.requests = []
        self.responses = iter(responses) if responses is not None else None
        self.supported_efforts = supported_efforts
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="topic-local")]))

    def create(self, **request):
        if self.supported_efforts is not None and request["reasoning_effort"] not in self.supported_efforts:
            raise ValueError("reasoning_effort não aceito pelo modelo simulado")
        self.requests.append(copy.deepcopy(request))
        if self.responses is not None:
            item = next(self.responses)
            if isinstance(item, Exception):
                raise item
            return item
        schema = request["response_format"]["json_schema"]["schema"]
        video_id = schema["properties"]["new_topics"]["items"]["properties"]["evidence_video_ids"]["items"]["enum"][0]
        # Cada lote falha uma vez: evidência duplicada não pode ser removida automaticamente.
        corrected = "ATENÇÃO: a saída anterior" in request["messages"][1]["content"]
        return response({"new_topics": [{
            "label": f"Tema {video_id}", "description": "Descrição do tema.",
            "evidence_video_ids": [video_id] if corrected else [video_id, video_id],
        }]})


class ComparableExperimentsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config_path = self.directory / "experiment_config.json"
        self.defaults = experiment.load_experiment_config(ROOT / "experiment_config.json")
        self.defaults["batch_size"] = 2
        self.config_path.write_text(json.dumps(self.defaults))
        self.records = [{"video_id": f"v{i}", "title": f"Título {i}"} for i in range(6)]
        self.input_path = self.directory / "geracao.jsonl"
        self.input_path.write_text("\n".join(json.dumps(r) for r in self.records))
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.stdout = redirect_stdout(io.StringIO())
        self.stderr = redirect_stderr(io.StringIO())
        self.stdout.__enter__()
        self.stderr.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.addCleanup(self.stderr.__exit__, None, None, None)

    def args(self, provider):
        return [
            "--provider", provider, "--config", str(self.config_path),
            "--input", str(self.input_path), "--output", str(self.directory / provider),
            "--experiment", "both",
        ]

    def cfg(self, provider="vllm"):
        return experiment.parse_args(self.args(provider))

    def test_notebook_and_cli_execute_identical_requests_and_orders(self):
        notebook = json.loads(NOTEBOOK.read_text())
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        namespace = {}
        for cell in cells[:-1]:
            exec(compile(cell, str(NOTEBOOK), "exec"), namespace)
        namespace.update(INPUT_FILE=self.input_path, OUTPUT_DIR=self.directory / "openai", CONFIG_FILE=self.config_path)
        openai_client = ControlledClient()
        with patch.object(experiment, "create_client", return_value=openai_client):
            exec(compile(cells[-1], str(NOTEBOOK), "exec"), namespace)
        openai_report = namespace["report"]

        local_client = ControlledClient(supported_efforts={"none", "high"})
        with patch.object(experiment, "create_client", return_value=local_client):
            self.assertEqual(experiment.main(self.args("vllm")), 0)
        local_report_path = self.directory / "vllm" / "summary_all.json"
        local_report = json.loads(local_report_path.read_text())

        self.assertEqual(len(openai_client.requests), 12)
        self.assertEqual(len(local_client.requests), 12)
        for left, right in zip(openai_client.requests, local_client.requests):
            self.assertEqual(left.pop("model"), "gpt-6.1-sol")
            self.assertEqual(right.pop("model"), "topic-local")
            self.assertEqual(left, right)
            self.assertEqual(left["reasoning_effort"], "high")
            self.assertNotIn("temperature", left)
            self.assertNotIn("top_p", left)

        self.assertEqual(openai_report["protocol_hash"], local_report["protocol_hash"])
        expected_orders = [self.records, list(self.records)]
        random.Random(self.defaults["shuffle_seed"]).shuffle(expected_orders[1])
        self.assertNotEqual(expected_orders[0], expected_orders[1])
        for index, (left, right) in enumerate(zip(openai_report["experiments"], local_report["experiments"])):
            self.assertEqual(left["comparison_hash"], right["comparison_hash"])
            self.assertEqual(left["total_failed_attempts"], 3)
            self.assertEqual(left["total_tokens"], 180)
            self.assertEqual(left["failed_total_tokens"], 90)
            self.assertEqual(left["successful_total_tokens"], 90)
            for report in (openai_report, local_report):
                folder = Path(report["run_directory"]) / left["experiment"]
                manifest = json.loads((folder / "order_manifest.json").read_text())
                self.assertEqual(manifest["video_ids_in_order"], [r["video_id"] for r in expected_orders[index]])
            openai_final = Path(openai_report["run_directory"]) / left["experiment"] / "topics_final.json"
            local_final = Path(local_report["run_directory"]) / right["experiment"] / "topics_final.json"
            self.assertEqual(json.loads(openai_final.read_text())["topics"], json.loads(local_final.read_text())["topics"])

    def test_hash_detects_changed_parameters_and_input(self):
        cfg = self.cfg()
        before = experiment.sha256_json(experiment.protocol_config(cfg))
        for key, value in [("reasoning_effort", "medium"), ("max_tokens", 4096), ("shuffle_seed", 9),
                           ("batch_size", 3), ("max_attempts", 5), ("sdk_max_retries", 0)]:
            changed = copy.copy(cfg)
            setattr(changed, key, value)
            self.assertNotEqual(before, experiment.sha256_json(experiment.protocol_config(changed)))
        changed_records = copy.deepcopy(self.records)
        changed_records[0]["title"] = "Outro título"
        self.assertNotEqual(experiment.sha256_json(self.records), experiment.sha256_json(changed_records))
        self.assertNotEqual(experiment.sha256_json(self.records), experiment.sha256_json(self.records[::-1]))

    def test_invalid_outputs_are_rejected_for_both_providers(self):
        batch = self.records[:2]
        existing = [{"label": "Conhecido", "description": "Um tema já identificado."}]
        topic = {"label": "Novo", "description": "Descrição.", "evidence_video_ids": ["v0"]}
        invalids = [
            {"new_topics": [{**topic, "label": " CONHECIDO "}]},
            {"new_topics": [topic, topic]},
            {"new_topics": [{**topic, "evidence_video_ids": ["v0", "v0"]}]},
            {"new_topics": [{**topic, "evidence_video_ids": ["fora-do-lote"]}]},
            {"new_topics": [{**topic, "evidence_video_ids": []}]},
            {"new_topics": [{**topic, "extra": True}]},
        ]
        for provider in ("openai", "vllm"):
            for invalid in invalids:
                with self.subTest(provider=provider, invalid=invalid):
                    client = ControlledClient([response(invalid), response({"new_topics": []})])
                    result, _, failures, _ = experiment.call_with_retries(
                        client, self.cfg(provider), existing, batch, 1, self.directory / "logs",
                    )
                    self.assertEqual(result, {"new_topics": []})
                    self.assertEqual(len(failures), 1)
                    self.assertIn(failures[0]["error"], client.requests[1]["messages"][1]["content"])
                    self.assertEqual(failures[0]["usage"]["total_tokens"], 30)

    def test_truncation_and_missing_usage_trigger_retry_even_for_valid_json(self):
        for invalid in [response({"new_topics": []}, finish_reason="length"),
                        response({"new_topics": []}, usage=False)]:
            client = ControlledClient([invalid, response({"new_topics": []})])
            result, _, failures, _ = experiment.call_with_retries(
                client, self.cfg(), [], self.records[:2], 1, self.directory / "logs",
            )
            self.assertEqual(result, {"new_topics": []})
            self.assertEqual(len(failures), 1)
            self.assertEqual(client.requests[0]["max_completion_tokens"], client.requests[1]["max_completion_tokens"])
            self.assertIn('"new_topics": []', Path(failures[0]["log_file"]).read_text())

    def test_api_error_is_logged_and_propagated(self):
        client = ControlledClient([ConnectionError("API indisponível")])
        log_dir = self.directory / "logs"
        with self.assertRaises(ConnectionError):
            experiment.call_with_retries(client, self.cfg(), [], self.records[:2], 1, log_dir)
        self.assertEqual(len(client.requests), 1)
        self.assertIn("API_ERROR ConnectionError", next(log_dir.glob("*.txt")).read_text())

    def test_exhausted_retries_preserve_logs(self):
        client = ControlledClient([response({"wrong": []})] * self.defaults["max_attempts"])
        log_dir = self.directory / "logs"
        with self.assertRaisesRegex(RuntimeError, "falhou em 3 tentativas"):
            experiment.call_with_retries(client, self.cfg(), [], self.records[:2], 1, log_dir)
        self.assertEqual(len(list(log_dir.glob("*.txt"))), 3)

    def test_new_runs_overwrite_previous_results_and_start_with_empty_taxonomy(self):
        cfg = self.cfg("openai")
        cfg.experiment = "original"
        first_topic = {"new_topics": [{
            "label": "Tema antigo", "description": "Um tema da primeira execução.", "evidence_video_ids": ["v0"],
        }]}
        client = ControlledClient([response(first_topic)] + [response({"new_topics": []})] * 5)
        first = experiment.run(cfg, client=client)
        folder = cfg.output / "experiment_1_original_order"
        final_path = folder / "topics_final.json"
        self.assertEqual(json.loads(final_path.read_text())["total_topics"], 1)
        stale_log = folder / "logs" / "stale-log.txt"
        stale_batch = folder / "batches" / "batch_999.json"
        stale_log.write_text("registro antigo")
        stale_batch.write_text("{}")
        second = experiment.run(cfg, client=client)
        self.assertEqual(first["run_directory"], second["run_directory"])
        self.assertEqual(Path(second["run_directory"]), cfg.output)
        self.assertFalse(list(cfg.output.glob("run_*")))
        self.assertFalse(stale_log.exists())
        self.assertFalse(stale_batch.exists())
        self.assertEqual(json.loads(final_path.read_text())["total_topics"], 0)
        self.assertTrue((cfg.output / "summary_all.json").is_file())
        self.assertIn("(nenhum tópico ainda)", client.requests[3]["messages"][1]["content"])

    def test_cleanup_only_removes_selected_experiments_and_managed_metadata(self):
        names = {"original": "experiment_1_original_order", "shuffled": "experiment_2_shuffled_order"}
        for selection in ("both", "original", "shuffled"):
            with self.subTest(selection=selection):
                cfg = self.cfg()
                cfg.experiment = selection
                cfg.output = self.directory / selection
                for name in names.values():
                    folder = cfg.output / name
                    folder.mkdir(parents=True)
                    (folder / "old.txt").write_text("antigo")
                other_provider = cfg.output / "openai"
                other_provider.mkdir()
                (other_provider / "keep.txt").write_text("outra LLM")
                note = cfg.output / "notes.txt"
                note.write_text("anotações")
                for name in ("protocol.json", "summary_all.json"):
                    (cfg.output / name).write_text("{}")
                experiment.clear_previous_results(cfg)
                for order, name in names.items():
                    removed = selection == "both" or selection == order
                    self.assertEqual((cfg.output / name).exists(), not removed)
                self.assertEqual(note.read_text(), "anotações")
                self.assertEqual((other_provider / "keep.txt").read_text(), "outra LLM")
                self.assertFalse((cfg.output / "protocol.json").exists())
                self.assertFalse((cfg.output / "summary_all.json").exists())

    def test_cleanup_rejects_symlink_before_removing_any_experiment(self):
        cfg = self.cfg()
        original = cfg.output / "experiment_1_original_order"
        original.mkdir(parents=True)
        (original / "old.txt").write_text("antigo")
        target = self.directory / "unrelated"
        target.mkdir()
        (target / "keep.txt").write_text("preservar")
        (cfg.output / "experiment_2_shuffled_order").symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "link simbólico"):
            experiment.clear_previous_results(cfg)
        self.assertTrue((original / "old.txt").is_file())
        self.assertEqual((target / "keep.txt").read_text(), "preservar")

    def test_incompatible_mistral_effort_fails_before_api_call_or_output_cleanup(self):
        cfg = self.cfg()
        cfg.reasoning_effort = "low"
        folder = cfg.output / "experiment_1_original_order"
        folder.mkdir(parents=True)
        marker = folder / "old.txt"
        marker.write_text("preservar")
        client = ControlledClient()
        with patch.dict(os.environ, {"TOKENIZER_MODE": "mistral"}):
            with self.assertRaisesRegex(ValueError, "defina reasoning_effort='high'"):
                experiment.run(cfg, client=client)
        self.assertEqual(client.requests, [])
        self.assertTrue(marker.is_file())

    def test_unknown_reasoning_tokens_remain_unknown_in_report(self):
        item = response({"new_topics": []})
        del item.usage.completion_tokens_details
        usage = experiment.response_usage(item)
        self.assertIsNone(usage["reasoning_tokens"])
        path = self.directory / "metrics.csv"
        experiment.export_report([usage], path)
        with path.open(encoding="utf-8-sig") as f:
            self.assertEqual(next(csv.DictReader(f))["reasoning_tokens"], "")

    def test_endpoint_credentials_do_not_leak_across_providers(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://local-only/v1", "VLLM_API_KEY": "local-key"}):
            cfg = self.cfg("openai")
        self.assertEqual(cfg.base_url, "https://api.openai.com/v1")
        self.assertIsNone(cfg.api_key)

    def test_invalid_shared_config_is_rejected(self):
        for field, value in [("batch_size", 0), ("max_completion_tokens", True),
                             ("reasoning_effort", "none"), ("sdk_max_retries", -1),
                             ("request_timeout", float("nan"))]:
            with self.subTest(field=field, value=value):
                config = {**self.defaults, field: value}
                self.config_path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    experiment.load_experiment_config(self.config_path)

    def test_entrypoint_configures_local_reasoning_without_running_a_real_server(self):
        bin_dir = self.directory / "bin"
        bin_dir.mkdir()
        scripts = {
            "chmod": "#!/bin/bash\nexit 0\n",
            "vllm": '''#!/bin/bash
printf '%s\\n' "$@" > "$CAPTURE_DIRECTORY/server-args.txt"
printf '%s\\n' "$REASONING_PARSER" "$GENERATION_CONFIG" > "$CAPTURE_DIRECTORY/server-env.txt"
trap 'exit 0' TERM
while :; do sleep 0.1; done
''',
            "python3": '''#!/bin/bash
printf '%s\\n' "$@" > "$CAPTURE_DIRECTORY/generator-args.txt"
for attempt in {1..50}; do
    if [[ -f "$CAPTURE_DIRECTORY/server-env.txt" ]]; then exit 0; fi
    sleep 0.02
done
exit 1
''',
        }
        for name, script in scripts.items():
            path = bin_dir / name
            path.write_text(script)
            path.chmod(0o700)
        result = subprocess.run(
            ["/bin/bash", str(ROOT / "entrypoint.sh")],
            env={
                "PATH": f"{bin_dir}:/usr/bin:/bin", "CAPTURE_DIRECTORY": str(self.directory),
                "MODEL_ID": "mistralai/Ministral-3-14B-Reasoning-2512",
            },
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = (self.directory / "server-args.txt").read_text().splitlines()
        self.assertEqual(args[args.index("--reasoning-parser") + 1], "mistral")
        self.assertEqual(args[args.index("--generation-config") + 1], "vllm")
        self.assertEqual(args[args.index("--tokenizer-mode") + 1], "mistral")
        self.assertEqual((self.directory / "server-env.txt").read_text().splitlines(), ["mistral", "vllm"])
        self.assertEqual((self.directory / "generator-args.txt").read_text().splitlines()[-2:], ["--provider", "vllm"])


if __name__ == "__main__":
    unittest.main()
