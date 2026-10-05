import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from anarkali.diagnostics import development_controls, state_ablations


class DiagnosticsTests(unittest.TestCase):
    def row(self, group, target, reverse=False):
        candidates = [{'id': 'a', 'text': 'alpha'}, {'id': 'b', 'text': 'beta'}]
        if reverse:
            candidates.reverse()
            target = list(reversed(target))
        return {'workflow': 'w', 'case_id': group+'::q', 'source_group': group,
                'state': {}, 'candidates': candidates, 'target': target}

    def test_priors_map_ids_after_candidate_permutation(self):
        train = [self.row('train', [0.9, 0.1])]
        dev = [self.row('dev', [0.8, 0.2], reverse=True)]
        report = development_controls(train, dev)['controls']['all']
        self.assertEqual(report['hard_majority_accuracy'], 1)
        self.assertEqual(report['soft_prior_accuracy'], 1)
        self.assertGreater(report['soft_prior_ce'], report['teacher_entropy'])

    def test_page_specific_options_get_a_uniform_prior(self):
        train = [self.row('train', [0.9, 0.1])]
        dev = self.row('dev', [0.0, 1.0])
        dev['candidates'] = [{'id': 'p0', 'text': 'MRP [₹1,999]'}, {'id': 'p1', 'text': 'Sale [₹1,299]'}]
        report = development_controls(train, [dev])
        self.assertEqual(report['development_rows_without_training_schema'], 1)
        self.assertEqual(report['controls']['all']['hard_majority_accuracy'], 0.5)

    def test_leakage_and_nonfinite_targets_rejected(self):
        train = [self.row('same', [0.9, 0.1])]
        with self.assertRaises(ValueError):
            development_controls(train, copy.deepcopy(train))
        bad = self.row('dev', [float('nan'), 0.1])
        with self.assertRaises(ValueError):
            development_controls(train, [bad])

    def test_prior_does_not_consume_state(self):
        train = [self.row('train', [0.9, 0.1])]
        dev = [self.row('dev', [0.8, 0.2])]
        first = development_controls(train, dev)
        dev[0]['state'] = {'changed': 'unrelated input'}
        self.assertEqual(first, development_controls(train, dev))

    def test_state_shuffle_keeps_whole_sources_consistent(self):
        rows = []
        for i in range(3):
            first = self.row('source'+str(i), [0.8,0.2])
            first['state'] = {'source':i}
            second = copy.deepcopy(first)
            second['case_id'] += '2'
            rows.extend([first,second])
        ablated = state_ablations(rows, 9)
        for i in range(0, len(rows), 2):
            shuffled = ablated['shuffled_within_workflow']
            self.assertEqual(shuffled[i]['state'], shuffled[i+1]['state'])
            self.assertNotEqual(shuffled[i]['state'], rows[i]['state'])
            self.assertEqual(shuffled[i]['target'], rows[i]['target'])
            self.assertEqual(shuffled[i]['candidates'], rows[i]['candidates'])


if __name__ == '__main__':
    unittest.main()
