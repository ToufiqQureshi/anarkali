"""Exercise full trainer locally with tiny random weights, without network or test split."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class TrainingPipelineTests(unittest.TestCase):
    def test_every_architecture_trains_and_saves_without_final_test(self):
        try:
            from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace
            from tokenizers.processors import TemplateProcessing
        except ImportError:
            self.skipTest('optional encoder dependencies unavailable')
        spec = importlib.util.spec_from_file_location('training_fixture', ROOT/'scripts'/'train_anarkali.py')
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        raw = Tokenizer(WordLevel({'[PAD]':0,'[UNK]':1,'[CLS]':2,'[SEP]':3,
                                    'approve':4,'reject':5,'review':6,'invoice':7}, unk_token='[UNK]'))
        raw.pre_tokenizer = Whitespace()
        raw.post_processor = TemplateProcessing(single='[CLS] $A [SEP]',
                                                pair='[CLS] $A [SEP] $B:1 [SEP]:1',
                                                special_tokens=[('[CLS]',2),('[SEP]',3)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]',
                                            unk_token='[UNK]', cls_token='[CLS]', sep_token='[SEP]')
        def tiny_encoder(*args, **kwargs):
            return BertModel(BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1,
                                        num_attention_heads=4, intermediate_size=32,
                                        max_position_embeddings=128))
        def row(group, index):
            return {'case_id': group+'::q', 'source_group': group, 'workflow':'invoice',
                    'state': {'invoice': index}, 'question':'invoice',
                    'candidates':[{'id':'a','text':'approve'},{'id':'b','text':'reject'}],
                    'target':[0.8,0.2] if index % 2 else [0.2,0.8]}
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)/'data'
            data.mkdir()
            manifest = {'split_counts':{}}
            for split in ('train','development'):
                payload = ''.join(json.dumps(row(split+str(i), i))+'\n' for i in range(4))
                path = data/f'{split}.jsonl'
                path.write_text(payload, encoding='utf-8')
                manifest['split_counts'][split] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            (data/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            for architecture in ('set','joint','packed'):
                output = Path(tmp)/architecture
                argv = ['train', '--data', str(data), '--output', str(output), '--architecture', architecture,
                        '--epochs','1','--batch-size','2','--device','cpu','--state-ablation',
                        '--overfit-source-cases','2','--max-state-tokens','32','--max-question-tokens','16',
                        '--max-candidate-tokens','16','--joint-max-tokens','64','--packed-max-tokens','64']
                if architecture == 'packed':
                    argv += ['--target-power','1.5','--selection-metric','accuracy_then_ce']
                with (patch.object(sys,'argv',argv), patch('huggingface_hub.HfApi.model_info',return_value=SimpleNamespace(sha='fixture')),
                     patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer),
                     patch('transformers.AutoModel.from_pretrained',side_effect=tiny_encoder),
                     contextlib.redirect_stdout(io.StringIO())):
                    trainer.main()
                self.assertTrue((output/'best.pt').exists())
                history = json.loads((output/'training.json').read_text())
                self.assertEqual(history['run_config']['architecture'], architecture)
                self.assertEqual(history['run_config']['model_revision'], 'fixture')
                self.assertEqual(len(history['history']), 1)
                if architecture == 'packed':
                    self.assertEqual(history['run_config']['target_power'], 1.5)
                    self.assertEqual(history['run_config']['selection_metric'], 'accuracy_then_ce')
                    self.assertEqual(history['best_development_argmax_accuracy'],
                                     history['history'][0]['development_argmax_accuracy'])
                ablations = json.loads((output/'state-ablation.json').read_text())
                self.assertEqual(set(ablations), {'original','blank','shuffled_within_workflow'})
                self.assertFalse((output/'anarkali-test.jsonl').exists())

            # A second packed run continues from the first one's weights.
            first = Path(tmp)/'packed'/'best.pt'
            argv = ['train', '--data', str(data), '--output', str(Path(tmp)/'continued'), '--architecture', 'packed',
                    '--epochs', '1', '--batch-size', '2', '--device', 'cpu', '--packed-max-tokens', '64',
                    '--init-checkpoint', str(first)]
            with (patch.object(sys,'argv',argv), patch('huggingface_hub.HfApi.model_info',return_value=SimpleNamespace(sha='fixture')),
                 patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer),
                 patch('transformers.AutoModel.from_pretrained',side_effect=tiny_encoder),
                 contextlib.redirect_stdout(io.StringIO()) as log):
                trainer.main()
            self.assertIn('"init_checkpoint"', log.getvalue())
            continued = json.loads((Path(tmp)/'continued'/'training.json').read_text())
            self.assertEqual(continued['run_config']['init_checkpoint'], str(first))
            with (patch.object(sys,'argv',argv + ['--model', 'other/encoder']),
                 patch('huggingface_hub.HfApi.model_info',return_value=SimpleNamespace(sha='fixture')),
                 patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer),
                 patch('transformers.AutoModel.from_pretrained',side_effect=tiny_encoder),
                 contextlib.redirect_stdout(io.StringIO())):
                with self.assertRaises(ValueError):
                    trainer.main()




if __name__ == '__main__':
    unittest.main()
