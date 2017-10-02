#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os
import unittest
from scipy import linalg

from .. import auxfn
from ..model.io_triangle import IOTriangle as green
from . import test_tools

currentdir = os.path.join(os.getcwd(), "mea/tests")
   
class TestGFAux(unittest.TestCase):
    """ A class that implements tests for the Auxiliary green function class. If input is given 
        as a sE file of a cluster, then it builds an auxiliary GF (it can do so
        for real or complex frequencies. However, imaginary frequencies is implicit).
        However, if an auxiliary green function file is given, 
        it can extract the self energy (it is implicit then that the frequencies
        are on the real axis"""
    
    @classmethod
    def setUpClass(TestGFAux):
        print("\nIn testauxfn.\n")

    def test_init(self):
        """ """
        fin_sE_to = os.path.join(currentdir, "files/self_moy.dat")
        gf_aux = auxfn.GFAux(fin_sE_to=fin_sE_to)
        
        self.assertEqual(gf_aux.zn_col, 0)
        self.assertEqual(gf_aux.fin_sE_to, fin_sE_to)
        self.assertEqual(gf_aux.fout_sE_ctow, "self_ctow.dat")
        self.assertEqual(gf_aux.fout_gf_aux_to, "gf_aux_to.dat")
                
 
    def test_build_gfvec_aux(self):
        """ """
        fin_sE_to = os.path.join(currentdir, "files/self_short_moy.dat") 
        gf_aux = auxfn.GFAux(fin_sE_to=fin_sE_to)
        gf_aux2 = auxfn.GFAux(fin_sE_to=fin_sE_to, rm_sE_ifty=True)
        gf_aux.build_gfvec_aux()
        gf_aux2.build_gfvec_aux()
        

        zn_vec = np.array([5.235989e-002, 1.570800e-001, 2.6179900e-001, 1.5027299e+001])
        
        
        sE_t = np.array([
            [  7.29148945e+00 -1.25688297e+00j,   4.31791692e+00 -7.37366863e-01j,
                        -4.37618486e+00 +7.44382438e-01j,   6.02348575e+00 -9.90968651e-01j,
                        4.05606418e+00 -4.07146712e-01j,   2.59560959e-01 -3.97514439e-03j],
                        [  6.37534240e+00 -3.08656815e+00j ,  3.92112500e+00 -1.86394356e+00j,
                        -3.80134014e+00 +1.78496589e+00j  , 5.15813767e+00 -2.33078630e+00j,
                        3.66679027e+00 -9.42835945e-01j   ,2.35758219e-01 -9.62837241e-03j],
                        [  5.42233260e+00 -4.06248226e+00j ,  3.52542822e+00 -2.58309048e+00j,
                        -3.22922418e+00 +2.29211959e+00j ,  4.28316747e+00 -2.88633370e+00j,
                        3.28618116e+00 -1.15036779e+00j ,  2.02677966e-01 -1.32361189e-02j],
                       [  5.55747767e+00 -2.25696764e+00j ,  5.68107370e+00 -2.11065758e+00j,
                       -2.65143495e-01 +3.18872569e-02j ,  6.68961495e-02 +1.31238862e-02j,
                       7.97463733e-02 +5.27132295e-02j  , 3.05312358e-03 -2.81419793e-02j]
                       ]
                       )

        sEvec_c = np.zeros((zn_vec.shape[0], 4, 4), dtype=complex)
        sEvec_c[:, 0, 0] = sE_t[:, 0] ; sEvec_c[:, 0, 1] = sE_t[:, 2] ; sEvec_c[:, 0, 2] = sE_t[:, 3] ; sEvec_c[:, 0, 3] = sE_t[:, 2]
        sEvec_c[:, 1, 0] = sEvec_c[:, 0, 1]; sEvec_c[:, 1, 1] = sE_t[:, 1] ; sEvec_c[:, 1, 2] = sEvec_c[:, 0, 1] ; sEvec_c[:, 1, 3] = sE_t[:, 4]
        sEvec_c[:, 2, 0] = sEvec_c[:, 0, 2] ; sEvec_c[:, 2, 1] = sEvec_c[:, 1, 2] ; sEvec_c[:, 2, 2] = sEvec_c[:, 0, 0] ; sEvec_c[:, 2, 3] = sEvec_c[:, 0, 1]
        sEvec_c[:, 3, 0] = sEvec_c[:, 0, 3] ; sEvec_c[:, 3, 1] = sEvec_c[:, 1, 3] ; sEvec_c[:, 3, 2] = sEvec_c[:, 2, 3] ; sEvec_c[:, 3, 3] = sEvec_c[:, 1, 1]

        sEvec_ir = green().c_to_ir(sEvec_c)

        sE_ifty = green().read_green_infty(sEvec_c)
        sE_ifty_ir = green().read_green_infty(sEvec_ir)

     
        
        # now let us form the gf_aux
        gfvec_test_c = np.zeros((zn_vec.shape[0], 4, 4), dtype=complex)
        gfvec_test_ir = gfvec_test_c.copy()
        gfvec_test_ir2 = gfvec_test_c.copy()
        
        for (i, sE) in enumerate(sEvec_c.copy()):
            gfvec_test_c[i] = linalg.inv(1.0j*np.eye(4)*zn_vec[i] - sE)
            
        for (i, sE) in enumerate(sEvec_ir.copy()):    
            gfvec_test_ir[i] = linalg.inv(1.0j*np.eye(4)*zn_vec[i] - sE + sE_ifty_ir)
            gfvec_test_ir2[i] = linalg.inv(1.0j*np.eye(4)*zn_vec[i] -sE)

        try:
            np.testing.assert_allclose(gf_aux.zn_vec, zn_vec, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_aux.sEvec_c, sEvec_c, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_aux.sE_infty, sE_ifty, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_aux.sEvec_c[:, 3, 3], sE_t[:, 1], rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_aux.gfvec_aux_c.shape, gfvec_test_c.shape, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_aux.gfvec_aux_c, gfvec_test_c, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_aux2.gfvec_aux_ir, gfvec_test_ir, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_aux.gfvec_aux_ir, gfvec_test_ir2, rtol=1e-7, atol=1e-7)
        except AssertionError:
            self.fail("ayaya np.allclose failed at test_build_gf_aux")



    def test_run_acon(self):
        """ """
        
        fin_sE_to = os.path.join(currentdir, "files/self_moyb60U3n05.dat")
        gf_aux = auxfn.GFAux(fin_sE_to=fin_sE_to, rm_sE_ifty=False)
        gf_aux.build_gfvec_aux()
        
        gf_aux.run_acon(fin_OME_default=os.path.join(currentdir, "files/OME_default.dat"), \
                  fin_OME_other=os.path.join(currentdir, "files/OME_other.dat"), \
                  fin_OME_input=os.path.join(currentdir, "files/OME_input_test.dat")
                  )

        # gf_aux.get_sE_w_list() put this line in the next test
        Aw_manual_small_truncation = np.loadtxt(os.path.join(currentdir,"files/Aw_manual_small_truncation.dat"))
        w_n_manual = Aw_manual_small_truncation[:, 0]
        Aw_manual = np.delete(Aw_manual_small_truncation,0, axis=1)

        w_n =gf_aux.w_vec_list[0]
        Aw = gf_aux.Aw_t_list[0][:, 0][:, np.newaxis]
        # print("Aw.shape = ", Aw.shape)
        # print(Aw_manual.shape)


        try:
            np.testing.assert_allclose(w_n.shape, w_n_manual.shape)
            np.testing.assert_allclose(Aw.shape, Aw_manual.shape)
            test_tools.compare_arrays(w_n, w_n_manual, rprecision=10**-3, n_diff_max=0, zero_equivalent=10**-5)
            test_tools.compare_arrays(Aw, Aw_manual, rprecision=10**-3, n_diff_max=0, zero_equivalent=10**-5)
        except AssertionError:
            self.fail("ayaya np.allclose failed at test_build_gf_aux")   


  
        
    def test_get_sEvec_w(self):
        """ """
        #print("\n\n IN test_get_sE_w \n\n")
        fin_sE_to = os.path.join(currentdir, "files/self_moy.dat")
        gf_aux = auxfn.GFAux(fin_sE_to=fin_sE_to, rm_sE_ifty=False)
        gf_aux.build_gfvec_aux()
        
        gf_aux.run_acon(fin_OME_default=os.path.join(currentdir, "files/OME_default.dat"), \
                  fin_OME_other=os.path.join(currentdir, "files/OME_other.dat"), \
                  fin_OME_input=os.path.join(currentdir, "files/OME_input_get_sE.dat")
                  )
        
        gf_aux.get_sEvec_w_list()

        sE_w_to_test = np.loadtxt("self_ctow0.dat")
        sE_w_to_test_good = np.loadtxt(os.path.join(currentdir, "files/self_ctow_test_good.dat"))

        try:
            # print("SHAPEs in test_auxiliary = ", sE_w_to_test.shape, " ", sE_w_to_test_good.shape)
            arr1 = sE_w_to_test.flatten()
            arr2 = sE_w_to_test_good.flatten()
            for i in range(arr1.shape[0]):
                if abs(arr1[i]) > 10**-2.0:
                    tmp = abs(arr1[i] - arr2[i])/abs(arr1[i])
                    if tmp > 10**-2.0:
                        print(tmp)
            test_tools.compare_arrays(sE_w_to_test, sE_w_to_test_good, rprecision=10**-2, n_diff_max=2, zero_equivalent=10**-4)
            #np.testing.assert_allclose(sE_w_to_test, sE_w_to_test_good, rtol=1e-3)
        except AssertionError:
            self.fail("Ayaya, np.allclose failed at test_get_sE_w")




if __name__ == '__main__':
    unittest.main()                