from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/spatial'))
import unittest
import numpy as np
from membrane_model import shell,contain,pores,body_sites,can_move
class MembraneTests(unittest.TestCase):
    def setUp(self):
        self.volume=np.zeros((15,15,15),bool);self.volume[3:12,3:12,3:12]=True
    def test_closed_shell_contains_no_gap(self):
        from scipy.ndimage import binary_fill_holes
        self.assertTrue(np.array_equal(binary_fill_holes(shell(self.volume)),self.volume))
    def test_containment_calculates_real_membrane_change(self):
        body=np.array([[10,7,7],[11,7,7],[12,7,7]])
        v=contain(self.volume,body);m=shell(v)
        self.assertGreater(np.count_nonzero(v),np.count_nonzero(self.volume))
        self.assertFalse(m[tuple(body.T)].any());self.assertTrue(v[tuple(body.T)].all())
    def test_pore_removes_only_membrane_and_small_hole_blocks_body(self):
        m=shell(self.volume);contact=np.zeros_like(m);contact[11,7,7]=True
        opened=pores(m,contact,1)
        self.assertTrue((opened<=m).all());self.assertLess(opened.sum(),m.sum())
        offsets=np.array([[0,0,0],[0,2,0]])
        occupied=np.full(self.volume.shape,-1,np.int32)
        self.assertFalse(can_move(np.array([10,7,7]),offsets,np.array([1,0,0]),opened,occupied,0,self.volume,False))
    def test_geometry_union_and_atomic_collision_gate(self):
        b=body_sites((0,0,0),(0,1,0));self.assertEqual(len(b),47)
        occ=np.full((15,15,15),-1,np.int32);occ[7,7,7]=1
        m=np.zeros_like(occ,bool);v=np.ones_like(occ,bool);before=occ.copy()
        self.assertFalse(can_move(np.array([6,7,7]),np.array([[0,0,0]]),np.array([1,0,0]),m,occ,0,v,False))
        self.assertTrue(np.array_equal(occ,before))
    def test_released_body_cannot_reenter(self):
        occ=np.full(self.volume.shape,-1,np.int32);m=np.zeros_like(self.volume)
        self.assertFalse(can_move(np.array([12,7,7]),np.array([[0,0,0]]),np.array([-1,0,0]),m,occ,0,self.volume,True))
if __name__=='__main__':unittest.main(verbosity=2)
