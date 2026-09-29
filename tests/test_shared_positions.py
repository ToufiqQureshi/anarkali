import unittest

from anarkali.packing import pack_row, shared_option_positions


class Tokenizer:
    cls_token_id, sep_token_id, pad_token_id, model_max_length = 1, 2, 0, 128

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 20 for c in text]


def row(texts):
    return {"state": {"x": "state words"}, "question": "pick",
            "candidates": [{"id": t, "text": t} for t in texts], "target": [1 / len(texts)] * len(texts)}


class PositionTests(unittest.TestCase):
    def test_every_option_starts_at_the_same_position(self):
        ids, spans, _ = pack_row(row(["yes", "no", "maybe"]), Tokenizer(), 64)
        positions = shared_option_positions(len(ids), spans)
        self.assertEqual(len(positions), len(ids))
        prefix = spans[0][0]
        self.assertEqual(positions[:prefix], list(range(prefix)))
        self.assertEqual({positions[start] for start, _ in spans}, {prefix})
        for start, end in spans:
            self.assertEqual(positions[start:end + 1], list(range(prefix, prefix + end - start + 1)))

    def test_bad_spans_are_rejected(self):
        with self.assertRaises(ValueError):
            shared_option_positions(10, [(4, 6)])


try:
    import torch
    from transformers import BertConfig, BertModel, ModernBertConfig, ModernBertModel
    from anarkali.packed import PackedChoiceModel, collate_packed
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch and transformers not installed")
class InvarianceTests(unittest.TestCase):
    TEXTS = ["alpha", "bravo", "charlie", "delta"]
    ORDER = [2, 0, 3, 1]

    def scores(self, model, texts, shared):
        batch = collate_packed([row(texts)], Tokenizer(), 96, shared)
        with torch.no_grad():
            return model(*batch[:-1]).logits[0]

    def permutation_gap(self, model, shared):
        base = self.scores(model, self.TEXTS, shared)
        permuted = self.scores(model, [self.TEXTS[i] for i in self.ORDER], shared)
        return float((permuted - base[self.ORDER]).abs().max())

    def test_bert_is_order_invariant_with_shared_positions(self):
        torch.manual_seed(0)
        encoder = BertModel(BertConfig(vocab_size=24, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                       intermediate_size=64, max_position_embeddings=128))
        shared = PackedChoiceModel(encoder, dropout=0, shared_option_positions=True).eval()
        self.assertLess(self.permutation_gap(shared, True), 1e-5)
        plain = PackedChoiceModel(encoder, dropout=0).eval()
        self.assertGreater(self.permutation_gap(plain, False), 1e-4)  # absolute positions do see the order

    def test_modernbert_global_attention_is_invariant_and_local_attention_runs(self):
        def encoder(window):
            torch.manual_seed(0)
            return ModernBertModel(ModernBertConfig(
                vocab_size=24, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
                global_attn_every_n_layers=3, local_attention=window, max_position_embeddings=128,
                pad_token_id=0, cls_token_id=1, sep_token_id=2, bos_token_id=1, eos_token_id=2))
        # A local window wider than every sequence makes all layers global. (global_attn_every_n_layers=1
        # says the same, but transformers 5.8 rejects its rope config.)
        all_global = PackedChoiceModel(encoder(512), dropout=0, shared_option_positions=True).eval()
        self.assertLess(self.permutation_gap(all_global, True), 1e-4)
        # Ettin's layout: local sliding-window layers between global ones. Measuring the window in
        # position IDs keeps these order-blind too.
        ettin_like = PackedChoiceModel(encoder(8), dropout=0, shared_option_positions=True).eval()
        self.assertLess(self.permutation_gap(ettin_like, True), 1e-5)
        plain = PackedChoiceModel(encoder(8), dropout=0).eval()
        self.assertGreater(self.permutation_gap(plain, False), 1e-5)

    def test_padded_batches_stay_finite_and_match_single_rows(self):
        torch.manual_seed(0)
        from transformers import ModernBertConfig, ModernBertModel
        encoder = ModernBertModel(ModernBertConfig(
            vocab_size=24, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
            global_attn_every_n_layers=3, local_attention=8, max_position_embeddings=128,
            pad_token_id=0, cls_token_id=1, sep_token_id=2, bos_token_id=1, eos_token_id=2))
        model = PackedChoiceModel(encoder, dropout=0, shared_option_positions=True).eval()
        rows = [row(["alpha", "bravo"]), row(["charlie", "delta", "echo", "foxtrot"])]
        batch = collate_packed(rows, Tokenizer(), 96, True)
        with torch.no_grad():
            together = model(*batch[:-1]).logits
            alone = model(*collate_packed(rows[:1], Tokenizer(), 96, True)[:-1]).logits
        self.assertTrue(torch.isfinite(together[0, :2]).all() and torch.isfinite(together[1]).all())
        self.assertLess(float((together[0, :2] - alone[0]).abs().max()), 1e-4)

    def test_half_precision_stays_finite_on_padding(self):
        # fp16 is what the notebooks train in; the additive mask must not create -inf rows
        torch.manual_seed(0)
        encoder = ModernBertModel(ModernBertConfig(
            vocab_size=24, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
            global_attn_every_n_layers=3, local_attention=8, max_position_embeddings=128,
            pad_token_id=0, cls_token_id=1, sep_token_id=2, bos_token_id=1, eos_token_id=2))
        model = PackedChoiceModel(encoder, dropout=0, shared_option_positions=True).half().eval()
        ids, mask, _spans, positions, _ = collate_packed([row(["a", "bb"]), row(["ccc", "d", "e"])], Tokenizer(), 96, True)
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model.encode(ids, mask, positions)).all())

    def test_missing_positions_are_refused(self):
        encoder = BertModel(BertConfig(vocab_size=24, hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
                                       intermediate_size=32, max_position_embeddings=128))
        model = PackedChoiceModel(encoder, shared_option_positions=True)
        batch = collate_packed([row(["a", "b"])], Tokenizer(), 64, False)
        with self.assertRaises(ValueError):
            model(*batch[:-1])

    def test_engine_arrays_carry_positions(self):
        from anarkali.engine import _batch_arrays
        packed = [pack_row(row(["yes", "no"]), Tokenizer(), 64), pack_row(row(["a", "b", "c"]), Tokenizer(), 64)]
        plain = _batch_arrays(packed, 0)
        shared = _batch_arrays(packed, 0, True)
        self.assertEqual(len(plain), 3)
        self.assertEqual(len(shared), 4)
        self.assertEqual(shared[3].shape, shared[0].shape)
        ids, spans, _ = packed[1]
        self.assertEqual(list(shared[3][1, :len(ids)]), shared_option_positions(len(ids), spans))


if __name__ == "__main__":
    unittest.main()


try:
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
    HAVE_ONNX = HAVE_TORCH
except ImportError:
    HAVE_ONNX = False


@unittest.skipUnless(HAVE_ONNX, "onnx and onnxruntime not installed")
class OnnxExportTests(unittest.TestCase):
    def test_exported_graph_matches_torch_and_stays_order_invariant(self):
        import itertools
        import tempfile
        import numpy as np
        import onnxruntime as ort
        from transformers import ModernBertConfig, ModernBertModel
        from anarkali.engine import _batch_arrays

        torch.manual_seed(0)
        model = PackedChoiceModel(ModernBertModel(ModernBertConfig(
            vocab_size=24, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
            global_attn_every_n_layers=3, local_attention=8, max_position_embeddings=256,
            pad_token_id=0, cls_token_id=1, sep_token_id=2, bos_token_id=1, eos_token_id=2)),
            dropout=0, shared_option_positions=True).eval()

        class Graph(torch.nn.Module):  # the same wrapper scripts/export_onnx.py exports
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, input_ids, attention_mask, candidate_spans, position_ids):
                return self.inner.head(self.inner.encode(input_ids, attention_mask, position_ids), candidate_spans)

        names = ["input_ids", "attention_mask", "candidate_spans", "position_ids"]
        tok = Tokenizer()
        example = _batch_arrays([pack_row(row(["alpha", "bravo"]), tok, 96),
                                 pack_row(row(["charlie", "delta", "echo"]), tok, 96)], 0, True)
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/model.onnx"
            torch.onnx.export(Graph(model).eval(), tuple(torch.from_numpy(a) for a in example), path,
                              input_names=names, output_names=["logits"], opset_version=17, dynamo=False,
                              dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"},
                                            "candidate_spans": {0: "b", 1: "o", 2: "t"},
                                            "position_ids": {0: "b", 1: "t"}, "logits": {0: "b", 1: "o"}})
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

            def run(texts):
                arrays = _batch_arrays([pack_row(row(texts), tok, 96)], 0, True)
                return session.run(["logits"], dict(zip(names, arrays)))[0][0], arrays

            texts = ["foxtrot", "golf", "hotel", "india"]  # a shape the export never saw
            base, arrays = run(texts)
            with torch.no_grad():
                reference = Graph(model)(*[torch.from_numpy(a) for a in arrays]).numpy()[0]
            self.assertLess(float(np.abs(base - reference).max()), 1e-4)
            for order in itertools.permutations(range(4)):
                permuted, _ = run([texts[i] for i in order])
                self.assertLess(float(np.abs(permuted - base[list(order)]).max()), 1e-4)
