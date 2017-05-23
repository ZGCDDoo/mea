#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os
import unittest
from random import random


from . import test_tools
from ..model import periodize_nambu, nambu
currentdir = os.path.join(os.getcwd(), "mea/tests")

#@unittest.skip("TO implement.")
class TestPeriodizeNambu(unittest.TestCase):
    """ A class that implements tests for the Auxiliary green function class. If input is given 
        as a gf file of a cluster, then it builds an auxiliary GF (it can do so
        for real or complex frequencies. However, imaginary frequencies is implicit).
        However, if an auxiliary green function file is given, 
        it can extract the gflf energy (it is implicit then that the frequencies
        are on the real axis"""
    
    @classmethod
    def setUpClass(cls):
        print("\nIn test periodize_nambu.\n")
        cls.fin_gf_to = os.path.join(currentdir, "files/self_short_sc_moy.dat") 
        (cls.z_vec, cls.sEvec_c) = nambu.read_nambu_c(cls.fin_gf_to, zn_col=0)
        
        cls.modeln = periodize_nambu.ModelNambu(1.0, 0.4, 1.115, cls.z_vec, cls.sEvec_c)


    def test_Model(self):
        """ """

        self.assertAlmostEqual(self.modeln.t, 1.0)
        self.assertAlmostEqual(self.modeln.tp, 0.4)
        self.assertAlmostEqual(self.modeln.mu, 1.115)

        try:
            np.testing.assert_allclose(self.modeln.sEvec_c, self.sEvec_c)
            np.testing.assert_allclose(self.modeln.z_vec, self.z_vec)
        except AssertionError:
            self.fail("np all close failed at test_Model") 
            
    #@unittest.skip
    def test_t_value(self):
        """ """
        (kx , ky) = (random(), random())
        t_value = self.modeln.t_value(kx, ky)
        t_v_test = None
        # hop_test = periodize_nambu.hopping_test(kx, ky, 1.0, 0.4)

        # try:
        #     test_tools.compare_arrays(t_value.real, hop_test.real)
        #     test_tools.compare_arrays(t_value.imag, hop_test.imag)
        # except AssertionError:
        #     self.fail("np all close failed at test_t_value")            
           


    # def test_periodize_Akw(self):
    #     """ """
    #     t, tp = (1.0, 0.4)
    #     kx, ky = (0.122, -0.987)
    #     k = np.array([kx,ky])
    #     sE = np.random.rand(4, 4)
    #     sE = np.array(sE, dtype=complex)
    #     ww = random()
    #     mu = random()
    #     N_c = 4
    #     r_sites = np.array([[0.0, 0.0], [1.0,0.0], [1.0,1.0], [0.0,1.0]])
    #     gf_ktilde = linalg.inv((ww + mu)*np.eye(4) - periodize.t_value(kx, ky, t,tp) - sE)

    #     gf_w_lattice = 0.0 + 0.0j

    #     for i in range(N_c):
    #         for j in range(N_c):
    #             gf_w_lattice += 1/N_c * np.exp(-1.0j*np.dot(k, r_sites[i] - r_sites[j]) )*gf_ktilde[i, j]
        
    #     Akw = -2.0*gf_w_lattice.imag
    #     Akw_test = periodize.periodize_Akw(kx, ky, t ,tp, sE, ww, mu)
        
    #     try:
    #         test_tools.compare_arrays(Akw, Akw_test)
    #     except AssertionError:
    #          self.fail("np all close failed at test_periodize_Akw")         


    # def test_periodize_Gkw(self):
    #     """ """
    #     t, tp = (1.0, 0.4)
    #     kx, ky = (0.122, -0.987)
    #     k = np.array([kx,ky])
    #     sE = np.random.rand(4, 4)
    #     sE = np.array(sE, dtype=complex)
    #     ww = random()
    #     mu = random()
    #     model = periodize.Model(t, tp, mu, np.array([ww]), np.array([sE]))
    #     Akw = periodize.periodize_Akw(kx, ky, t ,tp, sE, ww, mu)
    #     Akw_test = -2.0*periodize.periodize_Gkz_vec(model, kx, ky).imag

    #     try:
    #         self.assertAlmostEqual(Akw, Akw_test[0])
    #     except AssertionError:
    #         self.fail("np all close failed at test_periodize_Gkw")   

    # @unittest.skip("Dont want to test this for now.")
    # def test_fermi_surface(self):
    #     """ """
    #     fin_gf_to = os.path.join(currentdir, "files/self_moy.dat") 
    #     (z_vec, sEvec_c) = green.read_green_c(fin_gf_to, zn_col=0)
    #     model = periodize.Model(1.0, 0.4, 1.115, z_vec, sEvec_c)
    #     periodize.fermi_surface(model, w_value=0.0)
         


if __name__ == "__main__":

    unittest.main()