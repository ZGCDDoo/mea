
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import unittest

import os
from .. import acon


currentdir = os.path.join(os.getcwd(), "mea/tests")
gf_t_file = os.path.join(currentdir, "files/greenMoy.dat")
fin_OME_default = os.path.join(currentdir, "files/OME_default.dat")
fin_OME_other = os.path.join(currentdir, "files/OME_other.dat")
fin_OME_input = os.path.join(currentdir, "files/OME_input_acon.dat" )

class TestAcon(unittest.TestCase):
    """ """
    
    @classmethod
    def setUpClass(TestAcon):
        print("\nIn TestAcon.\n")

    def test_init(self):
        """ """
        
        
        gf_t_tmp = np.loadtxt(gf_t_file)
        zn_vec = gf_t_tmp[:, 0]
        gf_t_tmp = np.delete(gf_t_tmp, 0, axis=1)
        gf_t = gf_t_tmp[:, ::2] + 1.0j*gf_t_tmp[:, 1::2]
        ac = acon.ACon(gf_t, zn_vec, fin_OME_default=fin_OME_default, fin_OME_input=fin_OME_input, fin_OME_other=fin_OME_other)

        gf_aux_t = np.zeros(gf_t.shape, dtype=complex)
        loc = [0, 1, 2, 3]
        non_loc = [[0, 1, 2, 3, 4]]
        for (i, num) in enumerate(loc):
            gf_aux_t[:, i] = gf_t[:, num]
        for (i, nums) in enumerate(non_loc):
            l = i + len(loc)
            N = 4.0
            gf_aux_t[:, l] += 1/N*(gf_t[:, nums[0]] + gf_t[:, nums[1]] + 
                                   gf_t[:, nums[2]] + gf_t[:, nums[3]]) + \
                                   2.0/N*gf_t[:, nums[4]]
        
        ac.build_gf_aux_t()
        try:
            np.testing.assert_allclose(ac.zn_vec, zn_vec , rtol=1e-7)
            np.testing.assert_allclose(ac.gf_t, gf_t, rtol=1e-7)
            #np.testing.assert_allclose(ac.gf_aux_t, gf_aux_t, rtol=1e-5)
            #print(ac.gf_aux_t - gf_aux_t)
        except AssertionError:
            self.fail("np.testing.assert_allclose failed at test_init")
        
        self.assertEqual(ac.fout_Aw_t, "Aw_t.dat")
                
                
    def test_acon_preprocess(self):

        gf_t_tmp = np.loadtxt(gf_t_file)
        zn_vec = gf_t_tmp[:, 0]
        gf_t_tmp = np.delete(gf_t_tmp, 0, axis=1)
        gf_t = gf_t_tmp[:, ::2] + 1.0j*gf_t_tmp[:, 1::2]
        ac = acon.ACon(gf_t, zn_vec, fin_OME_default=fin_OME_default, fin_OME_input=fin_OME_input, fin_OME_other=fin_OME_other)
        ac.acon_preprocess()
        ac.cleanup()

    def test_make_OME_file(self):

        gf_t_tmp = np.loadtxt(gf_t_file)
        zn_vec = gf_t_tmp[:, 0]
        gf_t_tmp = np.delete(gf_t_tmp, 0, axis=1)
        gf_t = gf_t_tmp[:, ::2] + 1.0j*gf_t_tmp[:, 1::2]
        ac = acon.ACon(gf_t, zn_vec, fin_OME_default=fin_OME_default, fin_OME_input=fin_OME_input, fin_OME_other=fin_OME_other)

        #The following alone is not a good enough test. Add meat to the bone
        ac.acon_preprocess()
        ac.make_OME_file(iter_OME_input = 0, iteration=0)
        ac.cleanup()

        #test with iteration != 0

    def test_build_gf_aux_t_and_build_Aw_t(self):
        """ """ 
        gf_t_tmp = np.loadtxt(gf_t_file)
        zn_vec = gf_t_tmp[:, 0]
        gf_t_tmp = np.delete(gf_t_tmp, 0, axis=1)
        gf_t = gf_t_tmp[:, ::2] + 1.0j*gf_t_tmp[:, 1::2]
        ac = acon.ACon(gf_t, zn_vec, fin_OME_default=fin_OME_default, fin_OME_input=fin_OME_input, fin_OME_other=fin_OME_other)

        # try to see if building gf_aux_t and
        # build_Aw_t are really inverse transformations
        ac.build_gf_aux_t()
        Aw_aux_t = ac.gf_aux_t.real
        Aw_t = ac.build_Aw_t(Aw_aux_t)
        
        gf_aux_t = gf_t.copy()
        # loc = [0, 1, 2, 3] ; non_loc = [[0, 1 , 2, 3, 4]]  (defaults) 
        gf_aux_t[:, :-1:] = gf_t[:, :-1:].copy()
        gf_aux_t[:, -1] = 1/4.0*(gf_t[:, 0] + gf_t[:, 1] + gf_t[:, 2] + \
                               gf_t[:, 3] + 2.0*gf_t[:, 4])
        try:
            #pass
            np.testing.assert_allclose(gf_aux_t, ac.gf_aux_t, rtol=1e-7)
            np.testing.assert_allclose(Aw_t, gf_t.real, rtol=1e-7)
        except AssertionError:
            self.fail("error at build_gf_aux") 
        
        # TO DO: Now test for superconductivity, i.e also substracting gf in the non_local part
        loc = [0, 1, 2, 3]; non_loc = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]
        non_loc_sign = [[1, 1, 1, 1, 1], [-1, 1, 1, -1, 1]]
        non_loc_to_conjugate = [[False]*5, [False, False, True, True, False]]
        gf_t_sc = np.zeros((gf_t.shape[0], gf_t.shape[1] + 1), dtype=complex)
        gf_t_sc[:, :-1:] = gf_t.copy()
        gf_t_sc[:, -1] = 0.5*(gf_t[:, -1].copy() + gf_t[:, -2].copy())
        
        ac = acon.ACon(gf_t_sc, zn_vec, fin_OME_default=fin_OME_default, fin_OME_input=fin_OME_input, fin_OME_other=fin_OME_other, loc=loc, non_loc=non_loc,
                        non_loc_sign=non_loc_sign, non_loc_to_conjugate=non_loc_to_conjugate)


        gf_aux_t = gf_t_sc.copy()
        gf_aux_t[:, :-2:] = gf_t_sc[:, :-2:].copy()
        gf_aux_t[:, -2] = 1/4.0*(gf_t_sc[:, 0] + gf_t_sc[:, 1] + gf_t_sc[:, 2] + gf_t_sc[:, 3] + 2.0*gf_t_sc[:, 4])
        gf_aux_t[:, -1] = 1/4.0*(-gf_t_sc[:, 0] + gf_t_sc[:, 1] + gf_t_sc[:, 2].conjugate() - gf_t_sc[:, 3].conjugate() + 2.0*gf_t_sc[:, 5])

        # try to see if building gf_aux_t and
        # build_Aw_t are really inverse transformations
        ac.build_gf_aux_t()
        Aw_aux_t = ac.gf_aux_t.real.copy()
        Aw_t = ac.build_Aw_t(Aw_aux_t)

        Aw_test = np.zeros(Aw_t.shape)
        Aw_test[:, :-2:] = gf_aux_t[:, :-2:].copy().real
        Aw_test[:, -2] = 2.0*(gf_aux_t[:, -2].real) - 1/2.0*np.real(gf_t_sc[:, 0] + gf_t_sc[:, 1] + gf_t_sc[:, 2] + gf_t_sc[:, 3])
        Aw_test[:, -1] = np.real(2.0*gf_aux_t[:, -1] - 
                        1.0/2.0*(-gf_t_sc[:, 0] + gf_t_sc[:, 1] - np.fliplr([gf_t_sc[:, 2].copy()])[0] + np.fliplr([gf_t_sc[:, 3].copy()])[0] ) ) 
        
        
        try:
            np.testing.assert_allclose(gf_aux_t, ac.gf_aux_t, rtol=1e-7)
            np.testing.assert_allclose(Aw_t, Aw_test, rtol=1e-7)
        except AssertionError:
            self.fail("error at build_gf_aux") 

        
    def test_acon(self):
        """ """
        
 
        gf_t_tmp = np.loadtxt(gf_t_file)
        zn_vec = gf_t_tmp[:, 0]
        gf_t_tmp = np.delete(gf_t_tmp, 0, axis=1)
        gf_t = gf_t_tmp[:, ::2] + 1.0j*gf_t_tmp[:, 1::2]
        
        #this is good only if we test all the gf in a cluster without doing auxiliary_fn
        for (i, gf) in enumerate(np.transpose(np.copy(gf_t))):
            if not np.alltrue(np.absolute(np.sign(gf.imag) - np.sign(gf.imag[0])) < 10**(-5.0)):
                # print("\nWarning, the sign of the imaginary part changes for gf # %s", i)
                # print("Taking the same sign as the first element for all the gf")
                # print("Analytic continutation could give wrong results \n")
                gf_t[:, i].imag = np.sign(gf[0].imag)*np.absolute(gf.imag)        
        
        gf_t = np.copy(gf_t[:, :2:])
        # print("gf_t.shape == ", gf_t.shape)
        #print("In tests acon ... \n\n\n\n")
        ac = acon.ACon(gf_t, zn_vec, loc=[0,1], non_loc=[], non_loc_sign=[], non_loc_to_conjugate=[], fin_OME_default=fin_OME_default, fin_OME_input=fin_OME_input, fin_OME_other=fin_OME_other)
        ac.acon()
        
        Aw_t_good0 = np.loadtxt(os.path.join(currentdir, "files/Aw_test_acon0.dat"))
        w_vec0 = Aw_t_good0[:, 0]
        Aw_t_good0 = np.delete(Aw_t_good0,0, axis=1)

        Aw_t_good1 = np.loadtxt(os.path.join(currentdir, "files/Aw_test_acon1.dat"))
        w_vec1 = Aw_t_good1[:, 0]
        Aw_t_good1 = np.delete(Aw_t_good1,0, axis=1)


        try:
            np.testing.assert_allclose(ac.zn_vec, zn_vec , rtol=1e-4)
            np.testing.assert_allclose(ac.gf_t, gf_t, rtol=1e-4)
            np.testing.assert_allclose(ac.w_vec_list[0], w_vec0, rtol=1e-4)           
            np.testing.assert_allclose(ac.Aw_t_list[0], Aw_t_good0, rtol=1e-4)
            np.testing.assert_allclose(ac.w_vec_list[1], w_vec1, rtol=1e-4)           
            np.testing.assert_allclose(ac.Aw_t_list[1], Aw_t_good1, rtol=1e-4)
        except AssertionError:
            self.fail("np.testing.assert_array_allclose failed at test_acon")
        
        self.assertEqual(ac.fout_Aw_t, "Aw_t.dat")


if __name__ == '__main__':
    unittest.main() 


