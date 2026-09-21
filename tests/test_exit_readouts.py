from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/spatial'))
import unittest
from exit_readouts import summarize_events

class ReadoutTests(unittest.TestCase):
    def test_inside_censored(self):
        r=summarize_events([10,11],[0,0],[False,False],10,12,11)
        self.assertIsNone(r['first_anchor_outside_time'])
        self.assertEqual(r['censor_status'],'body_right_censored')
    def test_anchor_before_tail(self):
        r=summarize_events([10,11,12],[0,23,47],[False,True,True],10,12,10)
        self.assertEqual(r['delta_s'],1)
        self.assertEqual(r['anchor_return_count'],0)
    def test_return_then_exit(self):
        r=summarize_events([10,11,12,13],[0,23,12,47],[False,True,False,True],10,13,10)
        self.assertEqual(r['anchor_return_count'],1)
        self.assertEqual(r['delta_s'],2)
    def test_birth_after_gate(self):
        r=summarize_events([20,21,22],[0,23,47],[False,True,True],20,22,10)
        self.assertEqual(r['post_open_s'],2)
        self.assertEqual(r['delta_over_post_open'],.5)
    def test_all_outside_at_start(self):
        r=summarize_events([10],[47],[True],10,10,10)
        self.assertEqual(r['delta_s'],0)
        self.assertIsNone(r['delta_over_post_open'])
    def test_body_implies_anchor(self):
        with self.assertRaises(ValueError):
            summarize_events([10],[47],[False],10,11,10)

if __name__=='__main__':unittest.main()
