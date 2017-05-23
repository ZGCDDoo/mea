#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os
import unittest
from random import random
from scipy.integrate import dblquad


from . import test_tools
from ..model import periodize_nambu, nambu
currentdir = os.path.join(os.getcwd(), "mea/tests")


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
        cls.fin_gf_to = os.path.join(currentdir, "files/self_stiffness.dat") 
        (cls.z_vec, cls.sEvec_c) = nambu.read_nambu_c(cls.fin_gf_to, zn_col=0)
        cls.z_vec = 1.0j*cls.z_vec
        cls.mu = 2.9006319222252026
        cls.beta = 60.0
        cls.modeln = periodize_nambu.ModelNambu(1.0, 0.4, cls.mu, cls.z_vec, cls.sEvec_c)
        #cls.sE = 

    def test_Model(self):
        """ """

        self.assertAlmostEqual(self.modeln.t, 1.0)
        self.assertAlmostEqual(self.modeln.tp, 0.4)
        self.assertAlmostEqual(self.modeln.mu, 2.90063192222)

        try:
            np.testing.assert_allclose(self.modeln.sEvec_c, self.sEvec_c)
            np.testing.assert_allclose(self.modeln.z_vec, self.z_vec)
        except AssertionError:
            self.fail("np all close failed at test_Model") 

       
    def test_t_value(self):
        """ """
        (kx , ky) = (1.1, 0.2)
        t_value = self.modeln.t_value(kx, ky)
        t_v_test = np.array([
            [0.0, -0.411499 + 0.808496j, -0.0572445 + 0.206201j, -1.92106 + 0.389418j],
            [-0.411499 + -0.808496j, 0.0, -1.92106 + 0.389418j, -0.133024 + -0.167631j],
            [-0.0572445 + -0.206201j, -1.92106 + -0.389418j, 0.0, -0.411499 + -0.808496j],
            [-1.92106 + -0.389418j, -0.133024 + 0.167631j, -0.411499 + 0.808496j, 0.0] 
            ], dtype=complex)
        # hop_test = periodize_nambu.hopping_test(kx, ky, 1.0, 0.4)
        zeros_np = np.zeros((4, 4), dtype=complex)
        try:
            np.testing.assert_allclose(t_value[:4:, :4:], t_v_test, rtol=1e-4)
            np.testing.assert_allclose(t_value[4::, 4::], -t_v_test, rtol=1e-4)
            np.testing.assert_allclose(t_value[:4:, 4::], zeros_np)
            np.testing.assert_allclose(t_value[4::, :4:], zeros_np)
        except AssertionError:
            self.fail("np all close failed at test_t_value")            
           

    def test_periodize_nambu(self):
        """ """

        nambu_per = self.modeln.periodize_nambu(-3.14, 0.0, 0)
        
        nambu_test = np.array([ 
                              [0.380643 - 0.104037j , 0.302518],
                              [0.302518 , -0.380643 - 0.104037j]  
                              ], dtype=complex) 
        
        #print("\nnambu_per= \n", nambu_per)
        try:
            #pass
            np.testing.assert_allclose(nambu_per, nambu_test, rtol=1e-4)
        except AssertionError:
            self.fail("failed at test periodize_nambu.")



    def test_build_gf_ktilde(self):
        """ """
        nambu_ktilde = self.modeln.build_gf_ktilde(-0.29, 1.10, 0)
        
        nambu_ktilde_good = np.array([ 
                                     [0.170909 + -0.0717512j, 0.109254 + 0.035049j, -0.146594 + 0.0830717j, 0.133523 + -0.00971182j, 0.138348 + 3.62366e-17j, -0.174474 + -0.0531149j, -0.136188 + -0.0358539j, 0.176324 + -0.040396j], 

                                    [0.136638 + 0.0426773j, 0.163881 + -0.0636017j, 0.133523 + -0.00971182j, -0.05645 + 0.130343j, -0.174474 + 0.0531149j, 0.109947 + -2.97323e-18j, 0.172615 + -0.0433493j, -0.107064 + 0.0387002j], 

                                    [-0.141112 + 0.0410867j, 0.133743 + -0.0778787j, 0.170909 + -0.0717512j, 0.136638 + 0.0426773j, -0.136188 + 0.0358539j, 0.172615 + 0.0433493j, 0.131989 + 2.39876e-17j, -0.175463 + 0.0498096j], 

                                    [0.133743 + -0.0778787j, -0.1169 + -0.0423706j, 0.109254 + 0.035049j, 0.163881 + -0.0636017j, 0.176324 + 0.040396j, -0.107064 + -0.0387002j, -0.175463 + -0.0498096j, 0.111219 + 3.57149e-17j], 

                                    [0.131989 + 3.84044e-17j, -0.175463 + -0.0498096j, -0.136188 + -0.0358539j, 0.172615 + -0.0433493j, -0.170909 + -0.0717512j, -0.136638 + 0.0426773j, 0.141112 + 0.0410867j, -0.133743 + -0.0778787j], 

                                    [-0.175463 + 0.0498096j, 0.111219 + 2.6467e-17j, 0.176324 + -0.040396j, -0.107064 + 0.0387002j, -0.109254 + 0.035049j, -0.163881 + -0.0636017j, -0.133743 + -0.0778787j, 0.1169 + -0.0423706j], 

                                    [-0.136188 + 0.0358539j, 0.176324 + 0.040396j, 0.138348 + 3.99144e-17j, -0.174474 + 0.0531149j, 0.146594 + 0.0830717j, -0.133523 + -0.00971182j, -0.170909 + -0.0717512j, -0.109254 + 0.035049j], 

                                    [0.172615 + 0.0433493j, -0.107064 + -0.0387002j, -0.174474 + -0.0531149j, 0.109947 + 2.90373e-18j, -0.133523 + -0.00971182j, 0.05645 + 0.130343j, -0.136638 + 0.0426773j, -0.163881 + -0.0636017j]
                                    ])

        try:
            np.testing.assert_allclose(nambu_ktilde, nambu_ktilde_good, rtol=1e-4)
        except AssertionError:
            self.fail("failed at test periodize_nambu.")

    
    def test_stiffness(self):
        """ """
        y1 = self.modeln.Y1Limit
        y2 = self.modeln.Y2Limit

        stiffness = 2.0/self.beta * (2.0*np.pi)**(-2.0) * dblquad(self.modeln.stiffness, -np.pi, np.pi, y1, y2, args=(0,))[0]
        stiffness_good = 0.011908416
        self.assertAlmostEqual(stiffness, stiffness_good, places=5)



if __name__ == "__main__":

    unittest.main()