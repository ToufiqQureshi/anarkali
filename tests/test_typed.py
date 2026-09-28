import unittest

from anarkali.typed import concentration, format_answer, question_candidates


class TypedQuestionTests(unittest.TestCase):
    def test_choice_conversion(self):
        kind, text, candidates = question_candidates({
            "type": "choice",
            "instructions": "Pick a route.",
            "criteria": {"sales": "Sales lead.", "support": "Support request."},
        })
        self.assertEqual(kind, "choice")
        self.assertEqual(text, "Pick a route.")
        self.assertEqual(candidates, [{"id": "sales", "text": "Sales lead."},
                                      {"id": "support", "text": "Support request."}])

    def test_noul_conversion_defaults(self):
        kind, text, candidates = question_candidates({
            "type": "noul",
            "instructions": "The invoice is overdue.",
        })
        self.assertEqual(kind, "noul")
        self.assertEqual(text, "True or false: The invoice is overdue.")
        self.assertEqual([c["id"] for c in candidates], ["false", "true"])

    def test_score_conversion(self):
        kind, text, candidates = question_candidates({
            "type": "score",
            "instructions": "Estimate risk.",
            "criteria": ["low", "medium", "high"],
        })
        self.assertEqual(kind, "score")
        self.assertEqual(text, "Pick the level that fits best: Estimate risk.")
        self.assertEqual(candidates, [{"id": "0", "text": "low"}, {"id": "1", "text": "medium"},
                                      {"id": "2", "text": "high"}])

    def test_bad_input_errors(self):
        cases = [
            None,
            {},
            {"type": "unknown", "instructions": "x", "criteria": {"a": "A", "b": "B"}},
            {"type": "choice", "instructions": "x", "criteria": {"a": "A"}},
            {"type": "noul", "instructions": "x", "criteria": {"yes": "Y", "no": "N"}},
            {"type": "score", "instructions": "x", "criteria": ["only"]},
            {"type": "choice", "instructions": "x", "criteria": {"a": ""}},
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    question_candidates(case)

    def test_format_answer_shapes(self):
        candidates = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
        choice = format_answer("choice", candidates, [0.25, 0.75], abstain_below=0.8)
        self.assertEqual(choice["type"], "choice")
        self.assertEqual(choice["choice"], "b")
        self.assertEqual(choice["probabilities"], {"a": 0.25, "b": 0.75})
        self.assertTrue(choice["abstain"])

        noul = format_answer("noul", [{"id": "false", "text": "F"}, {"id": "true", "text": "T"}], [0.1, 0.9])
        self.assertEqual(noul["type"], "noul")
        self.assertEqual(noul["noul"], 0.9)
        self.assertEqual(noul["confidence"], 0.9)

        score = format_answer("score", [{"id": "0", "text": "low"}, {"id": "1", "text": "high"}], [0.4, 0.6])
        self.assertEqual(score["type"], "score")
        self.assertEqual(score["score"], 0.6)
        self.assertEqual(score["legend"], {"0": "low", "1": "high"})

    def test_concentration_matches_jev_formula(self):
        self.assertAlmostEqual(concentration([0.5, 0.5]), 0.0)
        self.assertAlmostEqual(concentration([1.0, 0.0]), 1.0)
        self.assertAlmostEqual(concentration([0.2, 0.3, 0.5]), 0.25)


if __name__ == "__main__":
    unittest.main()
