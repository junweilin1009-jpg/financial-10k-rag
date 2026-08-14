from __future__ import annotations

import csv
import hashlib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"


class DataManifestTests(unittest.TestCase):
    def test_bundled_filing_bytes_match_manifest(self) -> None:
        with MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {row["company"] for row in rows},
            {"Alphabet/Google", "Amazon", "Microsoft"},
        )
        for row in rows:
            path = DATA_DIR / row["file"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, int(row["bytes"]))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, row["sha256"], path)


if __name__ == "__main__":
    unittest.main()
