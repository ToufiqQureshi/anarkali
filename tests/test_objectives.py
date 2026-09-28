import random
import unittest

try:
    import torch
    from torch import nn
except ImportError:
    raise unittest.SkipTest("torch not installed (pip install '.[neural]')")

from anarkali.neural import HeadOutput, training_loss
from anarkali.objectives import (EMA, decision_losses, layerwise_groups, lr_lambda, ordinal_index,
                                 row_weights, shuffle_with_index, symmetric_kl, to_original_order)


def score_row(ids, target):
    return {"question_type": "score", "candidates": [{"id": i, "text": i} for i in ids], "target": target}


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(3, 4)
        self.mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        self.logits = self.logits.masked_fill(~self.mask, float("-inf"))
        self.targets = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.25, 0.25, 0], [0.9, 0.1, 0, 0]])

    def test_defaults_equal_training_loss(self):
        ours = decision_losses(self.logits, self.mask, self.targets)["loss"]
        theirs = training_loss(HeadOutput(self.logits, torch.zeros(3), self.mask), self.targets)["loss"]
        self.assertAlmostEqual(float(ours), float(theirs), places=6)

    def test_brier_matches_hand_computation(self):
        out = decision_losses(self.logits, self.mask, self.targets, brier_weight=1.0)
        probs = torch.softmax(self.logits, -1).nan_to_num(0)
        expected = ((probs - self.targets) ** 2).sum(-1).mean()
        self.assertAlmostEqual(float(out["brier"]), float(expected), places=5)
        self.assertGreater(float(out["loss"]), float(out["soft_ce"]))

    def test_rps_is_ordinal_and_order_independent(self):
        # levels arrive shuffled: ids 2,0,3,1; the teacher is sure of level 3
        rows = [score_row(["2", "0", "3", "1"], [0, 0, 1, 0])]
        index, ordinal = ordinal_index(rows, 4)
        self.assertEqual(index.tolist(), [[1, 3, 0, 2]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        target = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

        def rps_for(level_probs):  # level_probs in level order 0..3, placed at shuffled positions
            placed = torch.zeros(1, 4)
            for level, p in enumerate(level_probs):
                placed[0, index[0, level]] = p
            logits = placed.clamp_min(1e-9).log()
            return float(decision_losses(logits, mask, target, rps_weight=1.0, ordinal=(index, ordinal))["rps"])

        exact = rps_for([0.01, 0.01, 0.01, 0.97])
        near, far = rps_for([0.01, 0.01, 0.97, 0.01]), rps_for([0.97, 0.01, 0.01, 0.01])
        self.assertLess(exact, 0.01)
        self.assertLess(near, far)  # one level off costs less than three
        # cumulative predictions .97 .98 .99 against a teacher CDF of 0 0 0 1
        self.assertAlmostEqual(far, (0.97 ** 2 + 0.98 ** 2 + 0.99 ** 2) / 3, places=3)

    def test_non_score_rows_have_no_rps(self):
        rows = [{"candidates": [{"id": "a"}, {"id": "b"}], "target": [1, 0]}]
        index, ordinal = ordinal_index(rows, 2)
        out = decision_losses(torch.tensor([[0.0, 1.0]]), torch.ones(1, 2, dtype=torch.bool),
                              torch.tensor([[1.0, 0.0]]), rps_weight=5.0, ordinal=(index, ordinal))
        self.assertEqual(float(out["rps"]), 0.0)

    def test_weights(self):
        rows = [{"teacher_agreement": 1.0}, {"teacher_agreement": 0.0}, {}]
        w = row_weights(rows, "teacher_agreement")
        self.assertEqual(w.tolist(), [1.0, 0.10000000149011612, 1.0])
        self.assertIsNone(row_weights(rows, None))
        weighted = decision_losses(self.logits, self.mask, self.targets, weights=torch.tensor([1.0, 0.0, 0.0]))
        first = decision_losses(self.logits[:1], self.mask[:1], self.targets[:1])
        self.assertAlmostEqual(float(weighted["loss"]), float(first["loss"]), places=6)

    def test_gradients_flow_through_every_term(self):
        logits = self.logits.clone().requires_grad_(True)
        rows = [score_row(["0", "1", "2", "3"], [0.1, 0.2, 0.3, 0.4]),
                {"candidates": [{"id": x} for x in "abc"]}, {"candidates": [{"id": x} for x in "ab"]}]
        out = decision_losses(logits, self.mask, self.targets, brier_weight=0.5, rps_weight=0.5,
                              ordinal=ordinal_index(rows, 4))
        out["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad[self.mask]).all())
        self.assertEqual(float(logits.grad[~self.mask].abs().sum()), 0.0)


class ConsistencyTests(unittest.TestCase):
    def test_shuffle_round_trip(self):
        rows = [{"candidates": [{"id": x} for x in "abcd"], "target": [0.1, 0.2, 0.3, 0.4]},
                {"candidates": [{"id": x} for x in "ab"], "target": [0.6, 0.4]}]
        shuffled, index = shuffle_with_index(rows, random.Random(3))
        probs = torch.zeros(2, 4)
        for i, row in enumerate(shuffled):
            probs[i, :len(row["target"])] = torch.tensor(row["target"])
        back = to_original_order(probs, index)
        self.assertTrue(torch.allclose(back[0], torch.tensor([0.1, 0.2, 0.3, 0.4])))
        self.assertTrue(torch.allclose(back[1, :2], torch.tensor([0.6, 0.4])))
        self.assertEqual([c["id"] for c in rows[0]["candidates"]], list("abcd"))  # input untouched

    def test_symmetric_kl(self):
        mask = torch.ones(1, 3, dtype=torch.bool)
        p = torch.tensor([[0.2, 0.3, 0.5]])
        self.assertAlmostEqual(float(symmetric_kl(p, p, mask)), 0.0, places=6)
        q = torch.tensor([[0.5, 0.3, 0.2]])
        self.assertGreater(float(symmetric_kl(p, q, mask)), 0.1)
        self.assertAlmostEqual(float(symmetric_kl(p, q, mask)), float(symmetric_kl(q, p, mask)), places=6)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.embeddings = nn.Embedding(10, 4)
        self.encoder.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.encoder.final_norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 1)


class OptimiserTests(unittest.TestCase):
    def test_layerwise_groups(self):
        model = Toy()
        groups = layerwise_groups(model, 1e-4, 1e-3, 0.5)
        by_lr = {round(g["lr"], 12): g["params"] for g in groups}
        self.assertIn(1e-4, by_lr)                     # top layer and final norm
        self.assertIn(round(1e-4 * 0.25, 12), by_lr)   # bottom layer (two below the top)
        self.assertIn(round(1e-4 * 0.125, 12), by_lr)  # embeddings, below every layer
        self.assertEqual(groups[-1]["lr"], 1e-3)
        counted = sum(p.numel() for g in groups for p in g["params"])
        self.assertEqual(counted, sum(p.numel() for p in model.parameters()))
        self.assertEqual(len({id(p) for g in groups for p in g["params"]}), len(list(model.parameters())))

    def test_lr_lambda(self):
        f = lr_lambda(100, 0.1, "linear")
        self.assertAlmostEqual(f(0), 0.1)
        self.assertAlmostEqual(f(9), 1.0)
        self.assertAlmostEqual(f(55), 0.5)
        self.assertAlmostEqual(lr_lambda(100, 0.0, "cosine")(50), 0.5)
        self.assertEqual(lr_lambda(100, 0.0, "constant")(99), 1.0)
        with self.assertRaises(ValueError):
            lr_lambda(10, 0.1, "step")

    def test_ema_tracks_and_restores(self):
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)
        ema = EMA(model, 0.5)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update(model)
        ema.apply_to(model)
        self.assertTrue(torch.allclose(model.weight, torch.full((1, 2), 0.5)))
        ema.restore(model)
        self.assertTrue(torch.allclose(model.weight, torch.ones(1, 2)))
        with self.assertRaises(ValueError):
            EMA(model, 1.0)


if __name__ == "__main__":
    unittest.main()
