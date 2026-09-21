from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/spatial'))
import os,sys,unittest
import numpy as np
from walk import advance,seed_rng
from membrane_model import damage_update

class WalkTests(unittest.TestCase):
    def run_walk(self,membrane,volume,center,offsets,seed=123):
        c=np.array([center],np.int32);o=np.array([offsets],np.int32)
        occ=np.full(volume.shape,-1,np.int32);occ[tuple((c[0]+o[0]).T)]=0
        a=np.ones(1,bool);r=np.zeros(1,bool);rt=np.full(1,np.nan)
        seed_rng(seed);stats=advance(c,o,a,r,rt,membrane,volume,occ,0.,1.,100.)
        return c,r,rt,stats,occ
    def test_closed_membrane_prevents_escape(self):
        v=np.zeros((15,15,15),bool);v[3:12,3:12,3:12]=True
        from membrane_model import shell
        c,r,rt,stats,occ=self.run_walk(shell(v),v,[7,7,7],[[0,0,0]])
        self.assertFalse(r[0]);self.assertGreater(stats[2],0)
        self.assertEqual(np.count_nonzero(occ==0),1)
        self.assertEqual(stats[0],sum(stats[1:]))
    def test_full_object_exits_through_removed_membrane(self):
        v=np.zeros((15,15,15),bool);v[7,7,7]=True
        c,r,rt,stats,occ=self.run_walk(np.zeros_like(v),v,[7,7,7],[[0,0,0],[0,1,0]])
        self.assertTrue(r[0]);self.assertTrue(np.isfinite(rt[0]))
        self.assertFalse(v[tuple((c[0]+np.array([[0,0,0],[0,1,0]])).T)].any())
        self.assertEqual(np.count_nonzero(occ==0),2)
    def test_reproducibility(self):
        v=np.ones((15,15,15),bool);m=np.zeros_like(v)
        a=self.run_walk(m,v,[7,7,7],[[0,0,0]])
        b=self.run_walk(m,v,[7,7,7],[[0,0,0]])
        for i in (0,1,3,4):self.assertTrue(np.array_equal(a[i],b[i]))
    def test_capacity_and_damage_are_independent_breach_gates(self):
        cfg=dict(membrane_damage_per_new_displacement_site=.01,membrane_damage_per_internal_virion_s=.0005,
        membrane_breach_displacement_threshold=160,membrane_breach_capacity_failure_threshold=3,
        membrane_damage_breach_threshold=1.2,membrane_integrity_breach_threshold=.45,membrane_breach_min_internal_virions=0)
        a=damage_update(None,1,0,0,3,cfg,3)
        self.assertEqual(a['breach_reason'],'capacity_failure_threshold')
        b=damage_update(None,1,120,120,0,cfg,0)
        self.assertEqual(b['breach_reason'],'cumulative_damage_threshold')
if __name__=='__main__':unittest.main(verbosity=2)
