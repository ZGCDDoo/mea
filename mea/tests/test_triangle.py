#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os
import unittest
from scipy import linalg
from random import random


from . import test_tools
from ..model.io_triangle import IOTriangle as green
from ..model.triangle import Triangle as Model

currentdir = os.path.join(os.getcwd(), "mea/tests")



def build_model():
    """ """
    fin_gf_to = os.path.join(currentdir, "files/self_moy.dat") 
    (z_vec, sEvec_c) = green().read_green_c(fin_gf_to, zn_col=0)
    model = Model(1.0, 0.4, 0.0, 1.115, z_vec, sEvec_c)
    return model


class TestTriangle(unittest.TestCase):
    """ A class that implements tests for the Auxiliary green function class. If input is given 
        as a gf file of a cluster, then it builds an auxiliary GF (it can do so
        for real or complex frequencies. However, imaginary frequencies is implicit).
        However, if an auxiliary green function file is given, 
        it can extract the gflf energy (it is implicit then that the frequencies
        are on the real axis"""
    
    @classmethod
    def setUpClass(TestTriangle):
        print("\nIn test_triangle.\n")



    def test_init(self):
        """ """
        model = build_model()

        fin_gf_to = os.path.join(currentdir, "files/self_moy.dat") 
        (z_vec, sEvec_c) = green().read_green_c(fin_gf_to, zn_col=0)
        self.assertAlmostEqual(model.t, 1.0)
        self.assertAlmostEqual(model.tp, 0.4)
        self.assertAlmostEqual(model.tpp, 0.0)
        self.assertAlmostEqual(model.mu, 1.115)

        try:
            np.testing.assert_allclose(model.sEvec_c, sEvec_c)
            np.testing.assert_allclose(model.z_vec, z_vec)
        except AssertionError:
            self.fail("np all close failed at test_init") 
            

    def test_t_value(self):
        """ """

        model = build_model()
        kx , ky = (random(), random())
        t_value = model.t_value(kx, ky)
        hop_test = model.hopping_test(kx, ky)

        try:
            test_tools.compare_arrays(t_value.real, hop_test.real)
            test_tools.compare_arrays(t_value.imag, hop_test.imag)
        except AssertionError:
            self.fail("np all close failed at test_t_value")            


    def test_eps0(self):
        """ """

        model = build_model()
        kx, ky = (0.3, -0.19)
        eps0_value = model.eps_0(kx, ky)
        real_value = -4.66985
        self.assertAlmostEqual(eps0_value, real_value, places=4)

    
    def test_exp_k(self):
        """ """
        model = build_model()
        kx, ky = (0.3, -0.19)
        exp_k_value = model.exp_k(kx, ky)
        real_value = [1.0, 0.955336 + 0.29552j, 
                    0.993956 + 0.109778j, 0.982004 - 0.188859j]
        
        try:
            np.testing.assert_allclose(exp_k_value, real_value, rtol=1e-4)
        except AssertionError:
            self.fail("np all close failed at test_exp_k")  


    def test_periodize_Akw(self):
        """ """

        
        t, tp, tpp = (1.0, 0.4, 0.0)
        kx, ky = (0.122, -0.987)
        k = np.array([kx,ky])
        sE = np.random.rand(4, 4)
        sEarr = np.array([sE], dtype=complex)
        ww = random()
        mu = random()
        model = Model(t, tp, tpp, mu, np.array([ww]), sEarr)
        N_c = 4
        r_sites = np.array([[0.0, 0.0], [1.0,0.0], [1.0,1.0], [0.0,1.0]])
        gf_ktilde = linalg.inv((ww + mu)*np.eye(4) - model.t_value(kx, ky) - sE)

        gf_w_lattice = 0.0 + 0.0j

        for i in range(N_c):
            for j in range(N_c):
                gf_w_lattice += 1/N_c * np.exp(-1.0j*np.dot(k, r_sites[i] - r_sites[j]) )*gf_ktilde[i, j]
        
        Akw = -2.0*gf_w_lattice.imag
        Akw_test = model.periodize_Akw(kx, ky, 0)
        Akw2 = (-2.0*gf_w_lattice.imag)**(2.0)
        Akw2_test = (model.periodize_Akw2(kx, ky ,0))
        
        try:
            test_tools.compare_arrays(Akw, Akw_test)
            test_tools.compare_arrays(Akw2, Akw2_test)
            np.testing.assert_allclose(Akw, Akw_test)
            np.testing.assert_allclose(Akw2, Akw2_test)
        except AssertionError:
            self.fail("np all close failed at test_periodize_Akw")         



    def test_periodize_Gkz(self):
        """ """
        t, tp, tpp = (1.0, 0.4, 0.0)
        kx, ky = (0.122, -0.987)
        sE = np.random.rand(4, 4)
        sEarr = np.array([sE], dtype=complex)
        ww = random()
        mu = random()
        model = Model(t, tp, tpp,  mu, np.array([ww]), sEarr)
        Akw = model.periodize_Akw(kx, ky, 0)
        Akw_test = -2.0*model.periodize_Gkz_vec(kx, ky).imag

        try:
            self.assertAlmostEqual(Akw, Akw_test[0])
        except AssertionError:
            self.fail("np all close failed at test_periodize_Gkw")   


    @unittest.skip("Not testing test_fermi_surface for now.")
    def test_fermi_surface(self):
        """ """
        pass
        # fin_gf_to = os.path.join(currentdir, "files/self_moy.dat") 
        # (z_vec, sEvec_c) = green.read_green_c(fin_gf_to, zn_col=0)
        # model = periodize.Model(1.0, 0.4, 1.115, z_vec, sEvec_c)
        # periodize.fermi_surface(model, w_value=0.0)
         


if __name__ == "__main__":

    unittest.main()