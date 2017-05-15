#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os
import unittest

from ..model import green
from ..transport import sigmadc  

currentdir = os.path.join(os.getcwd(), "mea/tests")




#@unittest.skip("Still implementing TestSigmaDC")
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
        fin_gf_to = os.path.join(currentdir, "files/self_moy.dat")
        beta = 21.3
        (w_vec, sEvec_c) = green.read_green_c(fin_gf_to, zn_col=0)
        mu = 1.123
        cls.sdc = sigmadc.SigmaDC(w_vec, sEvec_c, beta, mu)



    def test_init(self):
        """ """
        sdc = self.sdc

        self.assertAlmostEqual(sdc.t, 1.0)
        self.assertAlmostEqual(sdc.tp, 0.4)
        self.assertAlmostEqual(sdc.mu, 1.123)
        self.assertAlmostEqual(sdc.prefactor, -2.0)

    
    def test_dfd_dw(self):
        """ """

        sdc = self.sdc
        self.assertAlmostEqual(sdc.dfd_dw(0.2), -0.292486, 5)
        self.assertAlmostEqual(sdc.dfd_dw(-0.1), -2.02208, 5)                     


if __name__ == "__main__":

    unittest.main()