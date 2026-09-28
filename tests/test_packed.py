import random
import unittest
import torch
from transformers import BertConfig, BertModel
from anarkali.packed import PackedChoiceModel, collate_packed, pack_row, shuffle_candidates
from anarkali.neural import training_loss


class Tokenizer:
    cls_token_id, sep_token_id, pad_token_id, model_max_length = 1, 2, 0, 128

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 20 for c in text]


def row(options=2):
    return {'state':{'x':'a'*100}, 'question':'pick',
            'candidates':[{'id':str(i), 'text':'yes' if i==0 else 'no'} for i in range(options)],
            'target':[1.0]+[0.0]*(options-1)}


class PackedTests(unittest.TestCase):
    def test_single_encoder_pass_gradients_and_padding(self):
        encoder = BertModel(BertConfig(vocab_size=24, hidden_size=16, num_hidden_layers=1,
                                       num_attention_heads=4, intermediate_size=32, max_position_embeddings=128))
        model = PackedChoiceModel(encoder, dropout=0).eval()
        batch = collate_packed([row(2),row(3)], Tokenizer(),64)
        calls = []
        handle = encoder.register_forward_hook(lambda m, a, o: calls.append(o.last_hidden_state.shape))
        output = model(*batch[:-1])
        handle.remove()
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],2)
        self.assertEqual(float(output.probabilities()[0,2]),0)
        training_loss(output,batch[-1])['loss'].backward()
        self.assertGreater(float(encoder.embeddings.word_embeddings.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.head.scorer[-1].weight.grad.abs().sum()),0)

    def test_schema_preserved_and_truncation_reported(self):
        tokens, spans, stats = pack_row(row(),Tokenizer(),64)
        self.assertEqual(len(tokens),64)
        self.assertGreater(stats['state_tokens_dropped'],0)
        for candidate, (start,end) in zip(row()['candidates'],spans):
            self.assertEqual(tokens[start:end],Tokenizer().encode(candidate['text']))
        bad = row()
        bad['question'] = 'x'*100
        with self.assertRaisesRegex(ValueError,'schema too long'):
            pack_row(bad,Tokenizer(),64)

    def test_shuffle_preserves_id_target_alignment_without_mutation(self):
        source = row(3)
        reordered = shuffle_candidates([source],random.Random(7))[0]
        original = dict(zip([c['id'] for c in source['candidates']],source['target']))
        actual = dict(zip([c['id'] for c in reordered['candidates']],reordered['target']))
        self.assertEqual(actual,original)
        self.assertEqual(source['candidates'][0]['id'],'0')


if __name__ == '__main__':
    unittest.main()
