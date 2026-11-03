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

    def write_taxonomy(self, name, topics):
        path = self.directory / name
        path.write_text(json.dumps({"total_topics": len(topics), "topics": topics}))
        return path

    def seed_topic(self, topic_id="T007", label="Tema inicial"):
        return {
            "topic_id": topic_id, "label": label, "description": "Descrição do tópico importado.",
            "evidence_video_ids": ["video-de-outra-execucao"], "created_in_batch": 17,
        }

    def generated_topic(self, label="Tema novo"):
        return {"new_topics": [{
            "label": label, "description": "Um tema sustentado pelos títulos do lote.",
            "evidence_video_ids": ["v0"],
        }]}

    def test_providers_share_processing_with_provider_specific_temperature(self):
        openai_client = ControlledClient()
        openai_report = experiment.run(self.cfg("openai"), client=openai_client)

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
            self.assertEqual(right.pop("temperature"), 0.0)
            self.assertEqual(left, right)
            self.assertEqual(left["reasoning_effort"], "high")
            self.assertNotIn("temperature", left)
            self.assertNotIn("top_p", left)

        self.assertNotEqual(openai_report["protocol_hash"], local_report["protocol_hash"])
        expected_orders = [self.records, list(self.records)]
        random.Random(self.defaults["shuffle_seed"]).shuffle(expected_orders[1])
        self.assertNotEqual(expected_orders[0], expected_orders[1])
        for index, (left, right) in enumerate(zip(openai_report["experiments"], local_report["experiments"])):
            self.assertNotEqual(left["comparison_hash"], right["comparison_hash"])
            self.assertEqual(left["input_hash"], right["input_hash"])
            self.assertNotIn("temperature", left["generation_parameters"])
            self.assertEqual(right["generation_parameters"]["temperature"], 0.0)
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

    def test_seeded_providers_preserve_independent_taxonomies_and_continue_ids(self):
        seeds = [
            [self.seed_topic("T002", "Original A"), self.seed_topic("T009", "Original B")],
            [self.seed_topic("T004", "Embaralhado")],
        ]
        paths = [self.write_taxonomy(f"seed-{index}.json", topics) for index, topics in enumerate(seeds)]
        cfg = experiment.parse_args(self.args("openai") + [
            "--initial-taxonomy-original", str(paths[0]), "--initial-taxonomy-shuffled", str(paths[1]),
        ])
        openai_client = ControlledClient()
        openai_report = experiment.run(cfg, client=openai_client)

        local_client = ControlledClient()
        with patch.object(experiment, "create_client", return_value=local_client):
            experiment.main(self.args("vllm") + [
                "--initial-taxonomy-original", str(paths[0]), "--initial-taxonomy-shuffled", str(paths[1]),
            ])
        local_report = json.loads((self.directory / "vllm" / "summary_all.json").read_text())
        self.assertEqual(len(openai_client.requests), 12)
        self.assertEqual(len(local_client.requests), 12)
        for left, right in zip(openai_client.requests, local_client.requests):
            self.assertNotIn("temperature", left)
            self.assertEqual(right["temperature"], 0.0)
            self.assertEqual({k: v for k, v in left.items() if k != "model"},
                             {k: v for k, v in right.items() if k not in ("model", "temperature")})
        for index, (name, next_id) in enumerate([
            ("experiment_1_original_order", "T010"), ("experiment_2_shuffled_order", "T005"),
        ]):
            summary = local_report["experiments"][index]
            self.assertNotEqual(summary["comparison_hash"], openai_report["experiments"][index]["comparison_hash"])
            self.assertEqual(summary["initial_taxonomy_hash"], openai_report["experiments"][index]["initial_taxonomy_hash"])
            self.assertEqual(summary["initial_topics"], len(seeds[index]))
            self.assertEqual(summary["generated_topics"], 3)
            for provider in ("openai", "vllm"):
                folder = self.directory / provider / name
                final = json.loads((folder / "topics_final.json").read_text())
                self.assertEqual(final["topics"][:len(seeds[index])], seeds[index])
                self.assertEqual(final["topics"][len(seeds[index])]["topic_id"], next_id)
                self.assertEqual(final["total_topics"], len(seeds[index]) + 3)
                snapshot = json.loads((folder / "initial_taxonomy.json").read_text())
                self.assertEqual(snapshot["topics"], seeds[index])
            prompt = openai_client.requests[index * 6]["messages"][1]["content"]
            for topic in seeds[index]:
                self.assertIn(f"{topic['topic_id']} | {topic['label']}: {topic['description']}", prompt)
            for topic in seeds[1 - index]:
                self.assertNotIn(topic["label"], prompt)

    def test_initial_taxonomy_validation_precedes_api_and_cleanup_for_all_selected_orders(self):
        cfg = self.cfg()
        cfg.initial_taxonomy_original = self.write_taxonomy("valid.json", [self.seed_topic()])
        cfg.initial_taxonomy_shuffled = self.directory / "invalid.json"
        folder = cfg.output / "experiment_1_original_order"
        folder.mkdir(parents=True)
        marker = folder / "old.txt"
        marker.write_text("preservar")
        seed = self.seed_topic()
        invalids = [
            [], {"new_topics": []}, {"topics": {}}, {"topics": [None]},
            {"topics": [{"label": "Sem campos de saída"}]},
            {"topics": [seed], "total_topics": 2},
            {"topics": [seed, {**seed, "label": "Outro rótulo"}]},
            {"topics": [seed, {**seed, "topic_id": "T008", "label": " TEMA INICIAL "}]},
        ]
        for key, value in [
            ("topic_id", "inválido"), ("topic_id", "T000"), ("label", " "), ("description", None),
            ("evidence_video_ids", []), ("evidence_video_ids", ["id", "id"]),
            ("evidence_video_ids", [5]), ("created_in_batch", True), ("created_in_batch", 0),
        ]:
            invalids.append({"topics": [{**seed, key: value}]})
        with patch.object(experiment, "create_client") as create_client:
            for invalid in invalids:
                with self.subTest(invalid=invalid):
                    cfg.initial_taxonomy_shuffled.write_text(json.dumps(invalid))
                    with self.assertRaisesRegex(ValueError, "Taxonomia inicial"):
                        experiment.run(cfg)
                    self.assertEqual(marker.read_text(), "preservar")
            cfg.initial_taxonomy_shuffled.write_text("{JSON inválido")
            with self.assertRaisesRegex(ValueError, "JSON inválido na taxonomia inicial"):
                experiment.run(cfg)
            cfg.initial_taxonomy_shuffled = self.directory / "missing.json"
            with self.assertRaises(FileNotFoundError):
                experiment.run(cfg)
            self.assertTrue(marker.exists())
            create_client.assert_not_called()

    def test_imported_labels_are_rejected_when_proposed_again(self):
        cfg = self.cfg("openai")
        cfg.experiment = "original"
        cfg.initial_taxonomy_original = self.write_taxonomy("seed.json", [self.seed_topic()])
        duplicate = {"new_topics": [{
            "label": " TEMA INICIAL ", "description": "Duplicado.", "evidence_video_ids": ["v0"],
        }]}
        client = ControlledClient([response(duplicate)] + [response({"new_topics": []})] * 3)
        report = experiment.run(cfg, client=client)
        self.assertEqual(report["experiments"][0]["total_topics"], 1)
        self.assertEqual(report["experiments"][0]["generated_topics"], 0)
        self.assertEqual(report["experiments"][0]["total_failed_attempts"], 1)
        self.assertIn("Tópico duplicado", client.requests[1]["messages"][1]["content"])

    def test_only_one_order_can_receive_an_initial_taxonomy(self):
        cfg = self.cfg("openai")
        cfg.initial_taxonomy_shuffled = self.write_taxonomy("seed.json", [self.seed_topic()])
        client = ControlledClient([response(self.generated_topic())] + [response({"new_topics": []})] * 5)
        report = experiment.run(cfg, client=client)
        self.assertEqual([summary["initial_topics"] for summary in report["experiments"]], [0, 1])
        self.assertIn("(nenhum tópico ainda)", client.requests[0]["messages"][1]["content"])
        self.assertIn("Tema inicial", client.requests[3]["messages"][1]["content"])

    def test_taxonomies_inside_outputs_are_loaded_before_either_order_is_cleaned(self):
        cfg = self.cfg("openai")
        path = cfg.output / "experiment_1_original_order" / "topics_final.json"
        path.parent.mkdir(parents=True)
        seed = self.seed_topic()
        path.write_text(json.dumps({"topics": [seed]}))
        cfg.initial_taxonomy_original = path
        cfg.initial_taxonomy_shuffled = path
        client = ControlledClient([response({"new_topics": []})] * 6)
        report = experiment.run(cfg, client=client)
        for summary in report["experiments"]:
            self.assertEqual(summary["initial_topics"], 1)
            self.assertEqual(summary["generated_topics"], 0)
            folder = cfg.output / summary["experiment"]
            self.assertEqual(json.loads((folder / "topics_final.json").read_text())["topics"], [seed])
            self.assertEqual(json.loads((folder / "initial_taxonomy.json").read_text())["topics"], [seed])
        # A saída pode ser importada novamente na execução seguinte.
        self.assertEqual(experiment.load_initial_taxonomy(path), [seed])

    def test_comparison_hash_tracks_taxonomy_contents_instead_of_path_or_source_model(self):
        cfg = self.cfg("openai")
        cfg.experiment = "original"
        paths = [self.write_taxonomy(f"seed-{index}.json", [self.seed_topic()]) for index in range(2)]
        paths[1].write_text(json.dumps({"model": "outra-llm", "topics": [self.seed_topic()]}))
        paths.append(self.write_taxonomy("changed.json", [self.seed_topic(label="Tema diferente")]))
        hashes = []
        for path in paths:
            cfg.initial_taxonomy_original = path
            report = experiment.run(cfg, client=ControlledClient([response({"new_topics": []})] * 3))
            hashes.append(report["experiments"][0]["comparison_hash"])
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[0], hashes[2])

    def test_environment_taxonomies_allow_cli_overrides_and_ignore_unselected_orders(self):
        seed = self.seed_topic()
        path = self.write_taxonomy("seed.json", [seed])
        empty = self.write_taxonomy("empty.json", [])
        with patch.dict(os.environ, {
            "INITIAL_TAXONOMY_ORIGINAL": str(path), "INITIAL_TAXONOMY_SHUFFLED": "missing.json",
        }):
            cfg = self.cfg("openai")
            overridden = experiment.parse_args(self.args("openai") + ["--initial-taxonomy-original", str(empty)])
        self.assertEqual(cfg.initial_taxonomy_original, path)
        self.assertEqual(overridden.initial_taxonomy_original, empty)
        for current, expected_count in [(cfg, 1), (overridden, 0)]:
            current.experiment = "original"
            client = ControlledClient([response(self.generated_topic())] + [response({"new_topics": []})] * 2)
            report = experiment.run(current, client=client)
            self.assertEqual(len(report["experiments"]), 1)
            self.assertEqual(report["experiments"][0]["initial_topics"], expected_count)

    def test_hash_detects_changed_parameters_and_input(self):
        cfg = self.cfg()
        before = experiment.sha256_json(experiment.protocol_config(cfg))
        for key, value in [("reasoning_effort", "medium"), ("max_tokens", 4096), ("shuffle_seed", 9),
                           ("batch_size", 3), ("max_attempts", 5), ("sdk_max_retries", 0), ("temperature", 0.7)]:
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
        result_expected = self.generated_topic()
        for invalid in [response(result_expected, finish_reason="length"),
                        response(result_expected, usage=False)]:
            client = ControlledClient([invalid, response(result_expected)])
            result, _, failures, _ = experiment.call_with_retries(
                client, self.cfg(), [], self.records[:2], 1, self.directory / "logs",
            )
            self.assertEqual(result, result_expected)
            self.assertEqual(len(failures), 1)
            self.assertEqual(client.requests[0]["max_completion_tokens"], client.requests[1]["max_completion_tokens"])
            self.assertIn(json.dumps(result_expected), Path(failures[0]["log_file"]).read_text())

    def test_markdown_json_is_parsed_and_raw_output_is_preserved(self):
        expected = self.generated_topic()
        body = json.dumps(expected)
        variants = [body, f"```{body}", f"```{body}```", f"```json\n{body}\n```", f"```JSON\n{body}"]
        for provider in ("openai", "vllm"):
            for index, raw in enumerate(variants):
                with self.subTest(provider=provider, raw=raw):
                    item = response(expected)
                    item.choices[0].message.content = raw
                    log_dir = self.directory / provider / f"fence-{index}"
                    result, _, failures, _ = experiment.call_with_retries(
                        ControlledClient([item]), self.cfg(provider), [], self.records[:2], 1, log_dir,
                    )
                    self.assertEqual(result, expected)
                    self.assertEqual(failures, [])
                    self.assertIn(f"=== OUTPUT ===\n{raw}\n", next(log_dir.glob("*.txt")).read_text())

    def test_empty_first_response_with_markdown_requires_a_topic_and_retries(self):
        expected = self.generated_topic()
        for provider in ("openai", "vllm"):
            with self.subTest(provider=provider):
                empty = response({"new_topics": []})
                empty.choices[0].message.content = '```{"new_topics": []}'
                client = ControlledClient([empty, response(expected)])
                result, _, failures, _ = experiment.call_with_retries(
                    client, self.cfg(provider), [], self.records[:2], 1, self.directory / provider,
                )
                self.assertEqual(result, expected)
                self.assertEqual(len(failures), 1)
                self.assertIn("gere pelo menos um tópico", failures[0]["error"])
                self.assertIn("pelo menos um tópico", client.requests[0]["messages"][1]["content"])
                schema = client.requests[0]["response_format"]["json_schema"]["schema"]
                self.assertEqual(schema["properties"]["new_topics"]["minItems"], 1)
        # Depois de importar ou gerar um tópico, não é necessário criar outro.
        existing = [self.seed_topic()]
        experiment.validate_result({"new_topics": []}, self.records[:2], existing)
        cfg = self.cfg()
        client = ControlledClient([response({"new_topics": []})])
        experiment.call_with_retries(client, cfg, existing, self.records[:2], 1, self.directory / "seeded")
        schema = client.requests[0]["response_format"]["json_schema"]["schema"]
        self.assertNotIn("minItems", schema["properties"]["new_topics"])

    def test_markdown_does_not_hide_non_json_text_or_invalid_evidence(self):
        for raw in ('Texto antes {"new_topics": []}', '```python\n{"new_topics": []}\n```',
                    '```{"new_topics": []}``` texto depois', '```{"new_topics": [', "```"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                experiment.parse_model_output(raw)
        invalid = self.generated_topic()
        invalid["new_topics"][0]["evidence_video_ids"] = ["fora-do-lote"]
        item = response(invalid)
        item.choices[0].message.content = f"```json\n{json.dumps(invalid)}\n```"
        client = ControlledClient([item, response(self.generated_topic())])
        _, _, failures, _ = experiment.call_with_retries(
            client, self.cfg(), [], self.records[:2], 1, self.directory / "invalid-evidence",
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("de outro lote", failures[0]["error"])

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
        second_topic = self.generated_topic(label="Tema da nova execução")
        client = ControlledClient(
            [response(first_topic)] + [response({"new_topics": []})] * 2
            + [response(second_topic)] + [response({"new_topics": []})] * 2,
        )
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
        final = json.loads(final_path.read_text())
        self.assertEqual(final["initial_topics"], 0)
        self.assertEqual(final["total_topics"], 1)
        self.assertEqual(final["topics"][0]["label"], "Tema da nova execução")
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
                             ("request_timeout", float("nan")), ("vllm_temperature", -0.1),
                             ("vllm_temperature", 2.1), ("vllm_temperature", True),
                             ("vllm_temperature", "0.0"), ("vllm_temperature", float("nan")),
                             ("vllm_temperature", float("inf"))]:
            with self.subTest(field=field, value=value):
                config = {**self.defaults, field: value}
                self.config_path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    experiment.load_experiment_config(self.config_path)

    def test_vllm_temperature_can_be_configured_and_overridden(self):
        self.assertEqual(self.cfg().temperature, 0.0)
        self.config_path.write_text(json.dumps({**self.defaults, "vllm_temperature": 0.25}))
        cfg = self.cfg()
        self.assertEqual(experiment.generation_parameters(cfg)["temperature"], 0.25)
        cfg = experiment.parse_args(self.args("vllm") + ["--temperature", "0.7"])
        request = experiment.completion_request(cfg, "prompt", self.records[:2])
        self.assertEqual(request["temperature"], 0.7)
        openai_cfg = self.cfg("openai")
        self.assertIsNone(openai_cfg.temperature)
        self.assertNotIn("temperature", experiment.generation_parameters(openai_cfg))
        # Configurações antigas continuam válidas e usam o padrão local 0.0.
        legacy = {key: value for key, value in self.defaults.items() if key != "vllm_temperature"}
        self.config_path.write_text(json.dumps(legacy))
        self.assertEqual(self.cfg().temperature, 0.0)

    def test_invalid_cli_temperatures_are_rejected(self):
        for value in ("-0.1", "2.1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                experiment.parse_args(self.args("vllm") + ["--temperature", value])
        with self.assertRaises(SystemExit):
            experiment.parse_args(self.args("openai") + ["--temperature", "0"])

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
        self.assertEqual(args[args.index("--attention-backend") + 1], "TRITON_ATTN")
        self.assertEqual((self.directory / "server-env.txt").read_text().splitlines(), ["mistral", "vllm"])
        self.assertEqual((self.directory / "generator-args.txt").read_text().splitlines()[-2:], ["--provider", "vllm"])


if __name__ == "__main__":
    unittest.main()
