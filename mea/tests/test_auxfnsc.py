#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os
import unittest
from scipy import linalg


from mea import  auxfnsc, auxfn
from mea.model import nambu
from mea.tests import test_tools


currentdir = os.path.join(os.getcwd(), "mea/tests")

#@unittest.skip("Skipping TestGFAuxSC because conventions in README.txt are not yet implemented.")
class TestGFAuxSC(unittest.TestCase):
    """ A class that implements tests for the Auxiliary green function class. If input is given 
        as a sE file of a cluster, then it builds an auxiliary GF (it can do so
        for real or complex frequencies. However, imaginary frequencies is implicit).
        However, if an auxiliary green function file is given, 
        it can extract the self energy (it is implicit then that the frequencies
        are on the real axis"""
    
    @classmethod
    def setUpClass(TestGFAuxSC):
        print("\nIn test_auxfnsc: \n")
    
    def __init__(self, methodName="runTest"):
        """ """
        super(TestGFAuxSC, self).__init__(methodName)

    def test_init(self):
        """ """
        fin_sE_to = os.path.join(currentdir, "files/self_moy_sc_b60tp04n0495U6500.dat")
        gf_aux = auxfnsc.GFAuxSC(fin_sE_to=fin_sE_to)
        
        self.assertEqual(gf_aux.zn_col, 0)
        self.assertEqual(gf_aux.fin_sE_to, fin_sE_to)
        self.assertEqual(gf_aux.fout_sE_ctow, "self_nambu_ctow.dat")
        self.assertEqual(gf_aux.fout_gf_aux_to, "gf_aux_sb.dat")
                
 
    def test_build_gfvec_aux(self):
        """ """

        fin_sE_to = os.path.join(currentdir, "files/self_moy_sc_b60tp04n0495U6500.dat")
        (zn_vec, sE_c) = nambu.read_nambu_c(fin_sE_to)
        gf_aux = auxfnsc.GFAuxSC(fin_sE_to=fin_sE_to)
        gf_aux.build_gfvec_aux()

        sEvec_ir = nambu.c_to_ir(sE_c)
        sEvec_sb = np.zeros((sEvec_ir.shape[0], 2, 2) , dtype=complex)
        sEvec_sb[:, 0, 0] = sEvec_ir[:, 2, 2].copy() ; sEvec_sb[:, 0, 1] = sEvec_ir[:, -1, 2].copy()
        sEvec_sb[:, 1, 0] = np.conjugate(np.transpose(sEvec_sb[:, 0, 1].copy())) ; sEvec_sb[:, 1, 1] = -np.conjugate(sEvec_ir[:, 3, 3])
        gfvec_aux_sb_test = np.zeros(sEvec_sb.shape, dtype=complex)

        for (i, sE) in enumerate(sEvec_sb):
            gfvec_aux_sb_test[i] = linalg.inv(1.0j*zn_vec[i]*np.eye(2, dtype=complex) - sE)

        try:
            np.testing.assert_allclose(sEvec_sb, gf_aux.sEvec_sb)
            np.testing.assert_allclose(gfvec_aux_sb_test, gf_aux.gfvec_aux_sb)
        except AssertionError:
            self.fail("Problem at test_build_gfvec_aux")


    def test_ac(self):
        """ """
        #pass
        fin_sE_to = os.path.join(currentdir, "files/self_moy_sc_b60tp04n0495U6500.dat")
        gf_aux = auxfnsc.GFAuxSC(fin_sE_to=fin_sE_to, rm_sE_ifty=False)
        gf_aux.build_gfvec_aux()
        
        gf_aux.ac(fin_OME_default=os.path.join(currentdir, "files/OME_default.dat"), \
                  fin_OME_other=os.path.join(currentdir, "files/OME_other.dat"), \
                  fin_OME_input=os.path.join(currentdir, "files/OME_input_test.dat")
                  )

        # gf_aux.get_sEvec_w_list() put this line in the next test
        #Aw_manual_small_truncation = np.loadtxt(os.path.join(currentdir,"files/Aw_manual_small_truncation.dat"))
        #w_n_manual = Aw_manual_small_truncation[:, 0]
        #Aw_manual = np.delete(Aw_manual_small_truncation,0, axis=1)

        #w_n =gf_aux.w_n_list[0]
        #Aw = gf_aux.Aw_t_list[0][:, 0][:, np.newaxis]
        # print("Aw.shape = ", Aw.shape)
        # print(Aw_manual.shape)


        try:
            pass
            #np.testing.assert_allclose(w_n.shape, w_n_manual.shape)
            #np.testing.assert_allclose(Aw.shape, Aw_manual.shape)
            #test_tools.compare_arrays(w_n, w_n_manual, rprecision=10**-2, n_diff_max=5, zero_equivalent=10**-5)
            #test_tools.compare_arrays(Aw, Aw_manual, rprecision=10**-2, n_diff_max=5, zero_equivalent=10**-5)
        except AssertionError:
            self.fail("ayaya np.allclose failed at test_build_gfvec_aux")   


  
        
    def test_get_sEvec_w(self):
        """ """
        pass




if __name__ == '__main__':
    unittest.main()                