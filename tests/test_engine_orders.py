import math
import unittest

from anarkali.engine import Engine, option_orders


class Tokenizer:
    cls_token_id, sep_token_id, pad_token_id, model_max_length = 1, 2, 0, 512

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 50 for c in text]


class PositionBiasedBackend:
    """Scores 'right' options high, and adds a bias to whatever option sits first."""

    def __init__(self, bias=2.0):
        self.tokenizer, self.bias, self.calls = Tokenizer(), bias, []

    def logits(self, packed):
        self.calls.append(len(packed))
        out = []
        for ids, spans, _ in packed:
            row = []
            for position, (start, end) in enumerate(spans):
                text = "".join(chr((i - 3) % 50 + 48) for i in ids[start:end])
                quality = 1.0 if text == "".join(chr((ord(c) % 50) + 48) for c in "right") else 0.0
                row.append(quality + (self.bias if position == 0 else 0.0))
            out.append(row)
        return out


QUESTIONS = {"pick": {"type": "choice", "instructions": "Pick the right option.",
                      "criteria": {"a": "wrong", "b": "right", "c": "meh"}}}
STATE = {"note": "x" * 80}


class OrderTests(unittest.TestCase):
    def test_offsets(self):
        self.assertEqual(option_orders(4, 1), [0])
        self.assertEqual(option_orders(4, 2), [0, 2])
        self.assertEqual(option_orders(3, 3), [0, 1, 2])
        self.assertEqual(option_orders(2, 5), [0, 1])
        with self.assertRaises(ValueError):
            option_orders(3, 0)

    def test_single_order_follows_the_position_bias(self):
        engine = Engine(PositionBiasedBackend(), name="t", max_tokens=256)
        answer = engine.predict(STATE, QUESTIONS)["answers"]["pick"]
        self.assertEqual(answer["choice"], "a")  # the biased first slot wins

    def test_averaging_removes_the_bias_and_maps_back_to_ids(self):
        engine = Engine(PositionBiasedBackend(), name="t", max_tokens=256, orders=3)
        answer = engine.predict(STATE, QUESTIONS)["answers"]["pick"]
        self.assertEqual(answer["choice"], "b")
        probs = answer["probabilities"]
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)
        # each option sat first exactly once, so the two wrong options tie exactly
        self.assertAlmostEqual(probs["a"], probs["c"], places=4)
        # a per-call override works too
        self.assertEqual(engine.predict(STATE, QUESTIONS, orders=1)["answers"]["pick"]["choice"], "a")

    def test_batches_are_capped(self):
        backend = PositionBiasedBackend()
        engine = Engine(backend, name="t", max_tokens=256, orders=3)
        engine.BATCH = 4
        questions = {f"q{i}": QUESTIONS["pick"] for i in range(3)}
        engine.predict(STATE, questions)
        self.assertEqual(backend.calls, [4, 4, 1])

    def test_temperature_by_type(self):
        backend = PositionBiasedBackend(bias=0.0)
        sharp = Engine(backend, name="t", max_tokens=256, temperature_by_type={"choice": 0.5})
        flat = Engine(backend, name="t", max_tokens=256, temperature_by_type={"noul": 0.5})
        p_sharp = sharp.predict(STATE, QUESTIONS)["answers"]["pick"]["probabilities"]["b"]
        p_flat = flat.predict(STATE, QUESTIONS)["answers"]["pick"]["probabilities"]["b"]
        expected_flat = math.exp(1) / (math.exp(1) + 2)
        self.assertAlmostEqual(p_flat, round(expected_flat, 4), places=4)
        self.assertGreater(p_sharp, p_flat)
        with self.assertRaises(ValueError):
            Engine(backend, name="t", max_tokens=256, temperature_by_type={"choice": 0})


if __name__ == "__main__":
    unittest.main()
