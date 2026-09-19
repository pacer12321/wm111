import ast
from pathlib import Path
import unittest

from prepare import instrument_op, instrument_transformer, RemoveObservers
from summarize_text import summarize_rank
import text_state_timer as timer


class TimingTests(unittest.TestCase):
    def test_original_ast_preserved_op(self):
        original = '''class BidirectionalLinearBranch:
    def forward_head_shard(self):
        earlier = 1
        text_slice = 2
        text_key = 3
        text_state = text_slice + text_key
        return text_state
'''
        result = instrument_op(original)
        self.assertEqual(ast.dump(ast.parse(original)), ast.dump(RemoveObservers().visit(ast.parse(result))))
        self.assertEqual(result.count('with _text_scope():'),1)

    def test_only_two_branch_calls_wrapped(self):
        original = '''class MiniMaxH3Attention:
    def _run_openvdn_ulysses(self):
        readout_head = self.linear_attention.forward_head_shard(1)
        source_readout_head = self.linear_attention.forward_head_shard(2)
        readout_head = readout_head + source_readout_head
        return readout_head
'''
        result = instrument_transformer(original)
        self.assertEqual(result.count('with _text_branch_scope('),2)
        self.assertIn('with _text_branch_scope("T",',result)
        self.assertIn('with _text_branch_scope("S",',result)

    def test_scope_reset_on_exception(self):
        with self.assertRaises(ValueError):
            with timer.branch_scope('S',25):
                self.assertEqual(timer._BRANCH,('S',25))
                raise ValueError()
        self.assertIsNone(timer._BRANCH)

    def test_one_copy_not_two_is_removable(self):
        records = [{'rank': 0,'world':4,'forward':step,'forward_npu_interval_ms':1000,
                    'text_calls':[{'side':s,'layer':i,'npu_interval_ms':1,'cpu_enqueue_seconds':.001}
                                  for i in range(50) for s in ('T','S')]} for step in range(1,50)]
        result=summarize_rank(records,100.)
        self.assertEqual(result['T']['calls'],2450)
        self.assertAlmostEqual(result['T_plus_S_interval_seconds'],4.9)
        self.assertAlmostEqual(result['second_copy_S_interval_seconds_NOT_guaranteed_savings'],2.45)

    def test_no_reserved_cards(self):
        text=(Path(__file__).parent/'launch_server.sh').read_text()
        self.assertIn('export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3\n',text)
        self.assertNotIn('0,1,2,3,4',text)


if __name__=='__main__':
    unittest.main()
