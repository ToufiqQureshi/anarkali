import itertools
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
    from torch import nn
except ImportError:
    raise unittest.SkipTest("torch not installed (pip install '.[neural]')")
from anarkali.neural import ChoiceHead, HeadConfig, training_loss
from anarkali.encoder import EncoderChoiceModel
from anarkali.smoke import run_smoke

torch.set_num_threads(1)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(40, 16)
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class NeuralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.head = ChoiceHead(HeadConfig(encoder_dim=16, hidden_dim=16, num_heads=4, dropout=0))
        self.head.eval()
        self.s = torch.randn(2, 6, 16)
        self.q = torch.randn(2, 3, 16)
        self.c = torch.randn(2, 4, 16)
        self.sm = torch.tensor([[1,1,1,1,0,0], [1,1,1,1,1,1]], dtype=torch.bool)
        self.qm = torch.ones(2, 3, dtype=torch.bool)
        self.cm = torch.tensor([[1,1,1,1], [1,1,0,0]], dtype=torch.bool)

    def predict(self, c=None, cm=None, s=None):
        return self.head(self.s if s is None else s, self.sm, self.q, self.qm,
                         self.c if c is None else c, self.cm if cm is None else cm)

    def test_all_candidate_permutations_preserve_semantic_scores(self):
        with torch.no_grad():
            reference = self.predict()
            for order in itertools.permutations(range(4)):
                index = torch.tensor(order)
                changed = self.predict(self.c[:, index], self.cm[:, index])
                torch.testing.assert_close(changed.probabilities(), reference.probabilities()[:, index], atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(changed.answerability_logits, reference.answerability_logits, atol=1e-6, rtol=1e-6)

    def test_padding_cannot_change_real_candidate_scores(self):
        with torch.no_grad():
            reference = self.predict()
            state = self.s.clone()
            candidates = self.c.clone()
            state[~self.sm] = float("nan")
            candidates[~self.cm] = float("nan")
            changed = self.predict(candidates, s=state)
            torch.testing.assert_close(changed.probabilities(), reference.probabilities())
            torch.testing.assert_close(changed.answerability_logits, reference.answerability_logits)
            self.assertEqual(changed.probabilities()[1, 2:].count_nonzero().item(), 0)

    def test_identical_candidate_representations_have_identical_scores(self):
        c = self.c[:, :1].expand(-1, 4, -1)
        cm = torch.ones(2, 4, dtype=torch.bool)
        with torch.no_grad():
            probabilities = self.predict(c, cm).probabilities()
        torch.testing.assert_close(probabilities, torch.full_like(probabilities, .25))

    def test_supported_and_unsupported_examples_train_without_nan(self):
        self.head.train()
        output = self.predict()
        target = torch.tensor([[0., 1., 0., 0.], [0., 0., 0., 0.]])
        losses = training_loss(output, target, answerable=torch.tensor([True, False]))
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]).item())
        for parameter in self.head.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
        self.assertGreater(self.head.scorer[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.head.answerability[-1].weight.grad.abs().sum().item(), 0)

    def test_zero_targets_need_explicit_unsupported_labels(self):
        output = self.predict()
        with self.assertRaises(ValueError):
            training_loss(output, torch.zeros(2, 4))
        losses = training_loss(output, torch.zeros(2, 4), answerable=torch.zeros(2, dtype=torch.bool))
        self.assertEqual(losses["choice_loss"].item(), 0)
        self.assertGreater(losses["answerability_loss"].item(), 0)

    def test_invalid_masks_and_target_mass_are_rejected(self):
        with self.assertRaises(ValueError):
            self.predict(cm=torch.zeros_like(self.cm))
        with self.assertRaises(ValueError):
            self.predict(cm=self.cm.long())
        target = torch.tensor([[1.,0.,0.,0.], [0.,0.,1.,0.]])
        with self.assertRaises(ValueError):
            training_loss(self.predict(), target)

    def test_extreme_temperature_keeps_probabilities_finite(self):
        output = self.predict()
        for temperature in (1e-300, .1, 1., 1e300):
            probabilities = output.probabilities(temperature)
            self.assertTrue(torch.isfinite(probabilities).all().item())
            torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))

    def test_overflowed_target_mass_is_rejected(self):
        target = torch.tensor([[3e38, 3e38, 0., 0.], [1., 0., 0., 0.]])
        with self.assertRaises(ValueError):
            training_loss(self.predict(), target)

    def test_pretrained_revision_must_be_explicit(self):
        with self.assertRaises(ValueError):
            EncoderChoiceModel.from_pretrained("example", revision="")

    def test_state_is_reused_and_padded_candidates_skip_encoder(self):
        encoder = TinyEncoder()
        model = EncoderChoiceModel(encoder, HeadConfig(encoder_dim=16, hidden_dim=16, num_heads=4, dropout=0))
        model.eval()
        state_ids = torch.randint(0, 40, (2, 6))
        q_ids = torch.randint(0, 40, (2, 3))
        c_ids = torch.randint(0, 40, (2, 4, 5))
        c_mask = self.cm.unsqueeze(-1).expand(-1, -1, 5)
        cached = model.encode_state(state_ids, self.sm)
        self.assertEqual(encoder.calls, 1)
        first = model.score_encoded(cached, q_ids, self.qm, c_ids, c_mask)
        model.score_encoded(cached, q_ids, self.qm, c_ids, c_mask)
        self.assertEqual(encoder.calls, 5)
        direct = model(state_ids, self.sm, q_ids, self.qm, c_ids, c_mask)
        torch.testing.assert_close(first.logits, direct.logits)
        torch.testing.assert_close(first.answerability_logits, direct.answerability_logits)
        target = torch.tensor([[1.,0.,0.,0.], [0.,1.,0.,0.]])
        training_loss(direct, target)["loss"].backward()
        self.assertGreater(encoder.embedding.weight.grad.abs().sum().item(), 0)

    def test_training_smoke_improves_only_its_synthetic_objective(self):
        result = run_smoke()
        self.assertTrue(result["passed"])
        self.assertLess(result["final_loss"], result["initial_loss"])
        self.assertFalse(result["semantic_benchmark"])
        self.assertFalse(result["superiority_established"])


if __name__ == "__main__":
    unittest.main()
