"""Valida as métricas e a execução do notebook sem baixar embeddings ou chamar APIs."""
import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "comparacao_taxonomias.ipynb"


class TaxonomyComparisonTest(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.cells = {cell["id"]: "".join(cell["source"]) for cell in self.notebook["cells"]}
        self.namespace = {}
        for cell_id in ("comparison-settings", "comparison-helpers", "consolidation-helpers"):
            exec(compile(self.cells[cell_id], str(NOTEBOOK), "exec"), self.namespace)
        self.a = [
            {"id": "T001", "label": "Tema A", "description": "Descrição A."},
            {"id": "T002", "label": "Tema B", "description": "Descrição B."},
        ]
        self.compare = self.namespace["comparar_topicos"]

    def test_permuted_topics_are_equivalent_without_matching_equal_ids(self):
        b = [dict(self.a[1], id="T001"), dict(self.a[0], id="T002")]
        cosine = self.namespace["matriz_cosseno"]([[2, 0], [0, 3]], [[0, 5], [7, 0]])
        report = self.compare(self.a, b, cosine)
        self.assertAlmostEqual(report["metrics"]["global_similarity"], 1)
        self.assertAlmostEqual(report["metrics"]["coverage_f1"], 1)
        self.assertEqual(report["best_matches"][0]["id_b"], "T002")
        self.assertEqual(report["best_matches"][2]["label_a"], "Tema B")

    def test_unequal_taxonomies_report_directional_coverage_and_missing_topics(self):
        a = self.a + [{"id": "T003", "label": "Extra", "description": "Tema adicional."}]
        matrix = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        report = self.compare(a, self.a, matrix)
        metrics = report["metrics"]
        self.assertAlmostEqual(metrics["similarity_a_to_b"], 2 / 3)
        self.assertAlmostEqual(metrics["similarity_b_to_a"], 1)
        self.assertAlmostEqual(metrics["global_similarity"], 5 / 6)
        self.assertAlmostEqual(metrics["coverage_a_to_b"], 2 / 3)
        self.assertAlmostEqual(metrics["coverage_b_to_a"], 1)
        self.assertAlmostEqual(metrics["coverage_f1"], 0.8)
        self.assertEqual(metrics["topics_a_below_threshold"], 1)
        self.assertEqual(metrics["topics_b_below_threshold"], 0)

    def test_global_metric_is_symmetric_and_threshold_only_changes_coverage(self):
        matrix = np.array([[0.7, 0.1], [0.2, 0.8]])
        forward = self.compare(self.a, self.a, matrix, 0.5)["metrics"]
        reverse = self.compare(self.a, self.a, matrix.T, 0.5)["metrics"]
        strict = self.compare(self.a, self.a, matrix, 0.9)["metrics"]
        self.assertAlmostEqual(forward["global_similarity"], 0.75)
        self.assertEqual(forward["global_similarity"], reverse["global_similarity"])
        self.assertEqual(forward["global_similarity"], strict["global_similarity"])
        self.assertEqual(forward["coverage_f1"], 1)
        self.assertEqual(strict["coverage_f1"], 0)

    def test_many_to_one_matches_do_not_implicitly_penalize_topic_count(self):
        report = self.compare(self.a, self.a[:1], [[1], [1]])
        self.assertEqual(report["metrics"]["global_similarity"], 1)
        self.assertEqual(report["metrics"]["topics_a"], 2)
        self.assertEqual(report["metrics"]["topics_b"], 1)

    def test_unrelated_topics_and_invalid_matrices(self):
        report = self.compare(self.a, self.a, [[0, -0.2], [-0.2, 0]])
        self.assertEqual(report["metrics"]["global_similarity"], 0)
        self.assertEqual(report["metrics"]["coverage_f1"], 0)
        for matrix in ([], [[1]], [[float("nan"), 0], [0, 1]], [[2, 0], [0, 1]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                self.compare(self.a, self.a, matrix)
        for threshold in (-0.1, 1.1, float("nan"), True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.compare(self.a, self.a, np.eye(2), threshold)
        for left, right in (([[0, 0]], [[1, 0]]), ([[1, 0]], [[1, 0, 0]])):
            with self.assertRaises(ValueError):
                self.namespace["matriz_cosseno"](left, right)

    def test_topics_and_consolidated_groups_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "taxonomy.json"
            load = self.namespace["carregar_taxonomia"]
            path.write_text(json.dumps({"groups": [
                {"topic_ids": ["T001", "T005"], "label": "Grupo", "description": "Descrição."}
            ]}), encoding="utf-8")
            self.assertEqual(load(path)[0]["id"], "G001")
            invalids = [{"topics": []}, {"topics": [{"topic_id": "T1", "label": "Sem descrição"}]},
                        {"topics": [{"topic_id": "T1", "label": "Tema", "description": "Descrição."}] * 2}]
            for data in invalids:
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(data=data), self.assertRaises(ValueError):
                    load(path)

    def test_notebook_exports_reports_with_controlled_embeddings(self):
        calls = []

        class ControlledEncoder:
            max_seq_length = 128

            def __init__(self, model_name, **kwargs):
                calls.append(model_name)

            def encode(self, texts, **kwargs):
                calls.append(texts)
                return np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)

        sentence_transformers = ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = ControlledEncoder
        ipython = ModuleType("IPython")
        display_module = ModuleType("IPython.display")
        display_module.HTML = lambda value: value
        displayed = []
        display_module.display = displayed.append
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = [directory / "a.json", directory / "b.json"]
            a = [{**topic, "topic_id": topic["id"]} for topic in self.a]
            a[0]["label"] = "<Tema A>"
            a.append({**a[0], "id": "T003", "topic_id": "T003", "label": "Variante A"})
            b = [dict(a[1], id="T001", topic_id="T001"), dict(a[0], id="T002", topic_id="T002")]
            b.append({**b[0], "id": "T003", "topic_id": "T003", "label": "Variante B"})
            paths[0].write_text(json.dumps({"topics": a}), encoding="utf-8")
            paths[1].write_text(json.dumps({"topics": b}), encoding="utf-8")
            original_inputs = [path.read_bytes() for path in paths]
            self.namespace.update(ARQUIVO_A=paths[0], ARQUIVO_B=paths[1], PASTA_SAIDA=directory / "reports")
            with patch.dict(sys.modules, {
                "sentence_transformers": sentence_transformers,
                "IPython": ipython, "IPython.display": display_module,
            }), redirect_stdout(io.StringIO()):
                exec(compile(self.cells["comparison-run"], str(NOTEBOOK), "exec"), self.namespace)
            report = json.loads((directory / "reports/comparison_report.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(report["metrics"]["global_similarity"], 1)
            self.assertEqual(report["metric_version"], "bidirectional_best_cosine_v1")
            self.assertEqual(len(report["best_matches"]), 4)
            self.assertEqual(report["embedding_max_sequence_length"], 128)
            self.assertEqual(report["workflow_version"], "internal_merge_then_external_comparison_v1")
            self.assertEqual(report["comparison_before_merge"]["metrics"]["topics_a"], 3)
            self.assertEqual(report["metrics"]["topics_a"], 2)
            self.assertEqual(report["internal_metrics"]["a"]["source_id_retention"], 1)
            self.assertAlmostEqual(report["internal_metrics"]["a"]["reduction_rate"], 1 / 3)
            for name, original in (("a", a), ("b", b)):
                consolidated = json.loads((directory / f"reports/consolidated_{name}.json").read_text())
                self.assertEqual(consolidated["source_document"]["topics"], original)
                ids = [id_ for group in consolidated["groups"] for id_ in group["topic_ids"]]
                self.assertEqual(sorted(ids), ["T001", "T002", "T003"])
                self.assertEqual(len(ids), len(set(ids)))
            with (directory / "reports/similarity_matrix.csv").open(encoding="utf-8-sig") as file:
                self.assertEqual(len(list(csv.reader(file))), 3)
            with (directory / "reports/best_matches.csv").open(encoding="utf-8-sig") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 4)
            rendered = (directory / "reports/comparison.html").read_text(encoding="utf-8")
            self.assertIn("&lt;Tema A&gt;", rendered)
            self.assertNotIn("<Tema A>", rendered)
            self.assertEqual(len(displayed), 2)
            self.assertEqual([path.read_bytes() for path in paths], original_inputs)
            self.assertEqual(len(calls[1]), 6)

    def test_internal_merge_preserves_members_and_uses_all_member_vectors(self):
        topics = self.a + [{"id": "T003", "label": "Outra forma do tema A", "description": "Variante A."}]
        vectors = np.array([[1, 0], [0, 1], [0.99, 0.1]])
        groups, centroids = self.namespace["consolidar_semanticamente"](topics, vectors, 0.85)
        self.assertEqual(len(groups), 2)
        index = next(i for i, g in enumerate(groups) if len(g["source_topic_ids"]) == 2)
        merged = groups[index]
        self.assertEqual(set(merged["source_topic_ids"]), {"T001", "T003"})
        self.assertGreaterEqual(merged["cohesion_min"], 0.85)
        unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        expected = unit[[0, 2]].mean(axis=0)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(centroids[index], expected)
        self.assertFalse(np.allclose(centroids[index], unit[0]))
        self.assertEqual({t["id"] for t in merged["source_topics"]}, {"T001", "T003"})

    def test_complete_link_blocks_transitive_chains(self):
        angles = np.deg2rad([0, 25, 50])
        vectors = np.column_stack([np.cos(angles), np.sin(angles)])
        topics = self.a + [{"id": "T003", "label": "Tema C", "description": "Descrição C."}]
        merge = self.namespace["consolidar_semanticamente"]
        groups, _ = merge(topics, vectors, 0.85)
        self.assertEqual(sorted(len(g["source_topic_ids"]) for g in groups), [1, 2])
        self.assertTrue(all(g["cohesion_min"] is None or g["cohesion_min"] >= 0.85 for g in groups))
        # Reordenar as linhas não muda os grupos quando IDs e textos são preservados.
        order = [2, 0, 1]
        reordered, _ = merge([topics[i] for i in order], vectors[order], 0.85)
        memberships = lambda values: {frozenset(g["source_topic_ids"]) for g in values}
        self.assertEqual(memberships(groups), memberships(reordered))

    def test_singletons_are_preserved_without_inventing_cohesion(self):
        merge = self.namespace["consolidar_semanticamente"]
        groups, centroids = merge(self.a, np.eye(2), 0.85)
        internal = self.namespace["metricas_internas"](self.a, groups, np.eye(2), centroids, 0.85, 0.5)
        self.assertEqual(internal["reduction_rate"], 0)
        self.assertEqual(internal["source_id_retention"], 1)
        self.assertEqual(internal["semantic_coverage_original_to_consolidated"], 1)
        self.assertIsNone(internal["merged_group_mean_cosine"])
        self.assertIsNone(internal["merged_group_min_cosine"])
        for threshold in (True, float("nan"), -0.1, 1.1):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                merge(self.a, np.eye(2), threshold)


if __name__ == "__main__":
    unittest.main()
