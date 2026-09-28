import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
    from torch import nn
except ImportError:
    raise unittest.SkipTest("torch not installed (pip install '.[neural]')")
from anarkali.joint import JointChoiceModel
from anarkali.neural import training_loss


class TinyContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.embedding = nn.Embedding(40, 16)
        self.seen_rows = 0

    def forward(self, input_ids, attention_mask):
        self.seen_rows = len(input_ids)
        x = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        context = (x * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        return SimpleNamespace(last_hidden_state=x+context)


class JointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(1)
        self.model = JointChoiceModel(TinyContextEncoder(), hidden_dim=16, dropout=0)
        self.ids = torch.tensor([[[1,2,3],[1,4,5],[0,0,0]], [[1,6,7],[1,8,9],[1,10,11]]])
        self.mask = self.ids != 0

    def test_padding_skipped_and_permutation_equivariant(self):
        self.model.eval()
        with torch.no_grad():
            first = self.model(self.ids, self.mask).probabilities()
            self.assertEqual(self.model.encoder.seen_rows, 5)
            order = torch.tensor([2,0,1])
            changed = self.model(self.ids[:, order], self.mask[:, order]).probabilities()
            torch.testing.assert_close(changed, first[:, order])
            self.assertEqual(first[0,2], 0)

    def test_joint_training_reaches_encoder_and_head(self):
        out = self.model(self.ids, self.mask)
        loss = training_loss(out, torch.tensor([[0.8,0.2,0.],[0.1,0.2,0.7]]))['loss']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for component in (self.model.encoder, self.model.head):
            grads = [p.grad for p in component.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)

    def test_checkpoint_roundtrip_preserves_predictions(self):
        import io
        buffer = io.BytesIO()
        self.model.eval()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        other = JointChoiceModel(TinyContextEncoder(), hidden_dim=16, dropout=0).eval()
        other.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(other(self.ids,self.mask).probabilities(),
                                       self.model(self.ids,self.mask).probabilities())


if __name__ == '__main__':
    unittest.main()
