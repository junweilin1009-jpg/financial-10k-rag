from __future__ import annotations

import unittest
from pathlib import Path

from evaluation.run_evaluation import (
    HOLDOUT_STAGE,
    load_questions,
    repository_state,
    require_holdout_acknowledgement,
)


class EvaluationSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = Path(__file__).resolve().parents[1] / "evaluation" / "question_bank.csv"

    def test_default_all_excludes_holdout(self) -> None:
        rows = load_questions(self.bank, requested=None, stage=None, limit=None)
        self.assertEqual(len(rows), 102)
        self.assertNotIn(HOLDOUT_STAGE, {row["evaluation_stage"] for row in rows})

    def test_stage_selects_only_holdout(self) -> None:
        rows = load_questions(self.bank, requested=None, stage=HOLDOUT_STAGE, limit=None)
        self.assertEqual(len(rows), 15)
        self.assertEqual({row["evaluation_stage"] for row in rows}, {HOLDOUT_STAGE})

    def test_holdout_requires_explicit_acknowledgement(self) -> None:
        rows = load_questions(self.bank, requested="HOLD-001", stage=None, limit=None)
        with self.assertRaisesRegex(ValueError, "Protected holdout"):
            require_holdout_acknowledgement(rows, acknowledged=False)
        require_holdout_acknowledgement(rows, acknowledged=True)

    def test_repository_state_records_current_commit(self) -> None:
        commit, dirty = repository_state()
        self.assertEqual(len(commit), 40)
        self.assertIsInstance(dirty, bool)


if __name__ == "__main__":
    unittest.main()
