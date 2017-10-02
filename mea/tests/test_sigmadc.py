#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os
import unittest

from ..model.io_triangle import IOTriangle as green
from ..model.triangle import Triangle
from ..transport import sigmadc  

currentdir = os.path.join(os.getcwd(), "mea/tests")



class TestSigmaDC(unittest.TestCase):
    """ A class that implements tests for the Auxiliary green function class. If input is given 
        as a gf file of a cluster, then it builds an auxiliary GF (it can do so
        for real or complex frequencies. However, imaginary frequencies is implicit).
        However, if an auxiliary green function file is given, 
        it can extract the gflf energy (it is implicit then that the frequencies
        are on the real axis"""
    

    @classmethod
    def setUpClass(cls):
        print("\nIn test_sigmadc.\n")
        fin_gf_to = os.path.join(currentdir, "files/self_ctow0_U6.25n0495b12.dat")
        beta = 12.0
        (w_vec, sEvec_c) = green().read_green_c(fin_gf_to, zn_col=0)
        mu = 3.1736422868580827
        model = Triangle(1.0, 0.4, 0.0, mu, w_vec, sEvec_c)
        cls.sdc = sigmadc.SigmaDC(model, beta)



    def test_init(self):
        """ """
        sdc = self.sdc
        model = self.sdc.model

        self.assertAlmostEqual(model.t, 1.0)
        self.assertAlmostEqual(model.tp, 0.4)
        self.assertAlmostEqual(model.mu, 3.1736422868580827)
        self.assertAlmostEqual(sdc.prefactor, 2.0)
        self.assertAlmostEqual(sdc.beta, 12.0)

    
    def test_dfd_dw(self):
        """ """

        sdc = self.sdc
        self.assertAlmostEqual(sdc.dfd_dw(0.2), -0.91506 , 5)
        self.assertAlmostEqual(sdc.dfd_dw(-0.1), -2.13473, 5)


    def test_calc_sigmadc(self):
        """ """
        self.sdc.cutoff = 8.0
        self.assertAlmostEqual(self.sdc.calc_sigmadc()[0, 0], 0.761333, 3)              


if __name__ == "__main__":

    unittest.main()