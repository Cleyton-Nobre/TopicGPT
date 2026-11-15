"""Verifica coerência dos dados e geração offline das figuras de classificação."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

import plot_classificacao as plots


class ClassificationPlotsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / "resultados"
        self.results.mkdir()
        for path in plots.DEFAULT_RESULTS.glob("*.csv"):
            shutil.copy2(path, self.results / path.name)

    def change_metrics(self, change):
        path = self.results / "metricas.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        change(rows)
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_loads_six_complete_conditions(self):
        results = plots.load_results(self.results)
        self.assertEqual(len(results.metrics), 6)
        self.assertEqual(len(results.sources), 8)
        self.assertEqual(results.classes, [f"C{i}" for i in range(9)])
        for key in plots.CONDITIONS:
            self.assertEqual(int(results.matrices[key].sum()), 291)

    def test_rejects_missing_condition(self):
        self.change_metrics(lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "seis condições"):
            plots.load_results(self.results)

    def test_rejects_duplicate_condition(self):
        self.change_metrics(lambda rows: rows.append(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicada"):
            plots.load_results(self.results)

    def test_rejects_incomplete_coverage(self):
        self.change_metrics(lambda rows: rows[0].update(classificados="290"))
        with self.assertRaisesRegex(ValueError, "cobertura incompleta"):
            plots.load_results(self.results)

    def test_rejects_nonfinite_score(self):
        self.change_metrics(lambda rows: rows[0].update(f1_macro="nan"))
        with self.assertRaisesRegex(ValueError, "fora do intervalo"):
            plots.load_results(self.results)

    def test_rejects_metrics_inconsistent_with_predictions(self):
        self.change_metrics(lambda rows: rows[0].update(acuracia="0.99"))
        with self.assertRaisesRegex(ValueError, "acurácia incompatível"):
            plots.load_results(self.results)

    def test_cli_renders_from_other_directory_without_modifying_inputs(self):
        before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.results.glob("*.csv")}
        output = self.root / "figures"
        process = subprocess.run([
            sys.executable, str(Path(plots.__file__).resolve()),
            "--results-dir", str(self.results), "--output-dir", str(output),
            "--formats", "png", "pdf", "svg", "--dpi", "60",
        ], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["artifacts"]), 21)
        self.assertEqual(len(manifest["inputs"]), 8)
        for artifact in manifest["artifacts"]:
            path = output / artifact["file"]
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])
            if path.suffix == ".png":
                with Image.open(path) as picture:
                    picture.verify()
        self.assertEqual(before, {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in self.results.glob("*.csv")})


if __name__ == "__main__":
    unittest.main()
