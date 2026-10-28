import unittest
from backend.import_huggingface import parse_grade, stage


class ImportTests(unittest.TestCase):
    def pair(self, grade="GEM MT 10"):
        return [dict(filename=f"123_{s}.jpg", cert_id="123", grade=grade) for s in ("front", "back")]

    def test_grades(self):
        self.assertEqual(parse_grade("NM-MT+ 8.5"), 8.5)
        for label in ("NM 7 (MC)", "N0: AUTHENTIC", "", "11", "MINT 99"):
            self.assertIsNone(parse_grade(label))

    def test_no_fabricated_labels_or_permission(self):
        rows, excluded, audit = stage(self.pair())
        self.assertEqual(rows[0]["labels"], {"grade": 10})
        self.assertFalse(rows[0]["training_allowed"])
        self.assertEqual(rows[0]["annotation_coverage"], "unknown")
        self.assertEqual(audit["candidate_physical_cards"], 1)
        self.assertFalse(excluded)

    def test_inconsistent_pair_excluded(self):
        rows = self.pair(); rows[0]["grade"] = "NM 7"
        self.assertFalse(stage(rows)[0])
        self.assertFalse(stage(rows[:1])[0])

    def test_invalid_paths_and_duplicates(self):
        rows = self.pair()
        with self.assertRaises(ValueError): stage(rows + rows)
        rows[0]["filename"] = "../123_front.jpg"
        with self.assertRaises(ValueError): stage(rows)
