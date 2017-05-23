#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Here every gf is a gfvec, i.e every green-function is a vector of green function.
To lighten the writing we supress the "vec". The same goes for the sE.
"""
import numpy as np
import os
import unittest
from ..model import green

currentdir = os.path.join(os.getcwd(), "mea/tests")


class TestGreen(unittest.TestCase):
    """ A class that implements tests for the Auxiliary green function class. If input is given 
        as a gf file of a cluster, then it builds an auxiliary GF (it can do so
        for real or complex frequencies. However, imaginary frequencies is implicit).
        However, if an auxiliary green function file is given, 
        it can extract the gflf energy (it is implicit then that the frequencies
        are on the real axis"""
    
    @classmethod
    def setUpClass(self):
        print("\nIn test green.\n")
        print("currentdir = ", currentdir, "\n")
    
    def test_read_green_c(self):
        """ """
        fin_gf_to = os.path.join(currentdir, "files/self_short_moy.dat") 
        (zn, gf_c) = green.read_green_c(fin_gf_to, zn_col=0)
        
        zn_test = np.array([5.235989e-002, 1.570800e-001, 2.6179900e-001, 1.5027299e+001])
        
        try:
            np.testing.assert_allclose(zn, zn_test , rtol=1e-7, atol=1e-7)
        except AssertionError:
            self.fail("np.testing.assert_array_allclose failed at test_read_green for zn")        
        
        gf_t = np.array([
                        [7.29148945e+00 -1.25688297e+00j, 4.31791692e+00 -7.37366863e-01j,
                        -4.37618486e+00 +7.44382438e-01j, 6.02348575e+00 -9.90968651e-01j,
                        4.05606418e+00 -4.07146712e-01j
                        ],
                        [6.37534240e+00 -3.08656815e+00j, 3.92112500e+00 -1.86394356e+00j,
                        -3.80134014e+00 +1.78496589e+00j, 5.15813767e+00 -2.33078630e+00j,
                        3.66679027e+00 -9.42835945e-01j
                        ],
                        [5.42233260e+00 -4.06248226e+00j, 3.52542822e+00 -2.58309048e+00j,
                        -3.22922418e+00 +2.29211959e+00j, 4.28316747e+00 -2.88633370e+00j,
                        3.28618116e+00 -1.15036779e+00j
                        ],
                        [5.55747767e+00 -2.25696764e+00j, 5.68107370e+00 -2.11065758e+00j,
                        -2.65143495e-01 +3.18872569e-02j, 6.68961495e-02 +1.31238862e-02j,
                        7.97463733e-02 +5.27132295e-02j
                        ]
                            ])
                                  
        gf_c_test = green.t_to_c(zn, gf_t)
        gf_t_test = green.c_to_t(zn, gf_c_test)
      
        try:
            np.testing.assert_allclose(gf_c, gf_c_test, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_t, gf_t_test, rtol=1e-7, atol=1e-7)
        except AssertionError:
            self.fail("np.testing.assert_allclose failed at test_read_green for gf_t") 

        # now test for gf_c
        gf_c_test = np.zeros((zn.shape[0], 4, 4), dtype=complex)
        gf_c_test[:, 0, 0] = gf_t[:, 0] ; gf_c_test[:, 0, 1] = gf_t[:, 2] ; gf_c_test[:, 0, 2] = gf_t[:, 3] ; gf_c_test[:, 0, 3] = gf_t[:, 2]
        gf_c_test[:, 1, 0] = gf_c_test[:, 0, 1]; gf_c_test[:, 1, 1] = gf_t[:, 1] ; gf_c_test[:, 1, 2] = gf_c_test[:, 0, 1] ; gf_c_test[:, 1, 3] = gf_t[:, 4]
        gf_c_test[:, 2, 0] = gf_c_test[:, 0, 2] ; gf_c_test[:, 2, 1] = gf_c_test[:, 1, 2] ; gf_c_test[:, 2, 2] = gf_c_test[:, 0, 0] ; gf_c_test[:, 2, 3] = gf_c_test[:, 0, 1]
        gf_c_test[:, 3, 0] = gf_c_test[:, 0, 3] ; gf_c_test[:, 3, 1] = gf_c_test[:, 1, 3] ; gf_c_test[:, 3, 2] = gf_c_test[:, 2, 3] ; gf_c_test[:, 3, 3] = gf_c_test[:, 1, 1]
        
        gf_c_test_02 = [6.02348575e+00 -9.90968651e-01j , 5.15813767e+00 -2.33078630e+00j, 4.28316747e+00 -2.88633370e+00j,
                   6.68961495e-02 +1.31238862e-02j]
        gf_c_test_00 = [ 7.29148945e+00 -1.25688297e+00j, 6.37534240e+00 -3.08656815e+00j,  5.42233260e+00 -4.06248226e+00j,
                    5.55747767e+00 -2.25696764e+00j]           
        
        try:
            np.testing.assert_allclose(gf_c, gf_c_test, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_c[:, 0, 2], gf_c_test_02, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_c[:, 0, 0], gf_c_test_00, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_c[:, 3, 3], gf_t_test[:, 1], rtol=1e-7, atol=1e-7)
        except AssertionError:
            self.fail("np.testing.assert_allclose failed at test_read_gflf for gf_c")
    
    
    def test_read_green_ir(self):
        """ """
        pass
            
          

    def test_read_gf_infty(self):
        """ """
        fin_gf_to = os.path.join(currentdir, "files/self_short_moy.dat") 
        (zn, gf_c) = green.read_green_c(fin_gf_to, 0)
        gf_infty = green.read_green_infty(gf_c)
        gf_test_infty = gf_c[-1, :, :]

        try:
            np.testing.assert_allclose(gf_infty, gf_test_infty, rtol=1e-7, atol=1e-7)
        except AssertionError:
             self.fail("np all close failed at test_read_infty")            

 
    def test_t_to_c(self):
        fin_gf_to = os.path.join(currentdir,"files/self_moy.dat")
        (zn, gf_c) = green.read_green_c(fin_gf_to, 0)

        gf_to_test = green.c_to_to(zn, gf_c)
        gf_t_test = green.c_to_t(zn, gf_c)
        
        gf_test1_c = green.to_to_c(zn, gf_to_test)
        gf_test2_c = green.t_to_c(zn, gf_t_test)

        try:
            np.testing.assert_allclose(gf_c, gf_test1_c, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_c, gf_test2_c, rtol=1e-7, atol=1e-7)
        except AssertionError:
            self.fail("Ayaya, np.testing.assert_allclose failed at test_t_to_c")    
        

    def test_c_to_ir(self):
        """ """
        fin_gf_to = os.path.join(currentdir,"files/self_moy.dat")
        (zn, gf_c) = green.read_green_c(fin_gf_to, 0)

        # test to see if going from c_to_ir and back gives the same thing
        gf_ir = green.c_to_ir(gf_c)
        gf_c_test = green.ir_to_c(gf_ir)
        
        gf_ir_test = np.zeros(gf_c.shape, dtype=complex)
        gf_ir_test[:, 0, 0] = gf_c[:, 0, 0] + gf_c[:, 0 ,2]
        gf_ir_test[:, 0, 1] = 2.0*gf_c[:, 0, 1]
        gf_ir_test[:, 1, 1] = gf_c[:, 1, 1] + gf_c[:, 1, 3]
        gf_ir_test[:, 2, 2] = gf_c[:, 0, 0] - gf_c[:, 0, 2]
        gf_ir_test[:, 3, 3] = gf_c[:, 1, 1] - gf_c[:, 1, 3]
        
    
        zeros = np.zeros((gf_ir.shape[0]))
        try:
            np.testing.assert_allclose(gf_c, gf_c_test,rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 0, 2], zeros, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 0, 3], zeros, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 1, 2], zeros, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 1, 3], zeros, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 2, 0], zeros, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 2, 1], zeros, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 2, 3], zeros, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 3, 0], zeros, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 3, 1], zeros, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 3, 2], zeros, rtol=1e-7, atol=1e-7)
            
            np.testing.assert_allclose(gf_ir[:, 0, 0], gf_ir_test[:, 0, 0], rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 0, 1], gf_ir_test[:, 0, 1], rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 1, 0], gf_ir_test[:, 0, 1], rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 1, 1], gf_ir_test[:, 1, 1], rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 2, 2], gf_ir_test[:, 2, 2], rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 3, 3], gf_ir_test[:, 3, 3], rtol=1e-7, atol=1e-7)
        except AssertionError:
            self.fail("Ayaya, np.testing.assert_allclose failed at test_t_to_c")  
    

    def test_t_to_ir(self):
        """test t_to_ir, to_to_ir and ir_to_to"""
    
        fin_gf_to = os.path.join(currentdir,"files/self_moy.dat")
        (zn, gf_c) = green.read_green_c(fin_gf_to, 0)

        # test to see if going from c_to_ir and back gives the same thing
        gf_ir = green.c_to_ir(gf_c)    
        gf_t = green.ir_to_t(zn, gf_ir)
        gf_to = green.ir_to_to(zn, gf_ir)
        gf_ir_test = green.to_to_ir(zn, gf_to)
        gf_t_test = green.to_to_t(gf_to)
        gf_ir_test2 = green.t_to_ir(zn, gf_t)
        
        try:
            np.testing.assert_allclose(gf_ir, gf_ir_test, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir, gf_ir_test2, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_t, gf_t_test, rtol=1e-7, atol=1e-7)
        except AssertionError:
            self.fail("Ayaya, np.testing.assert_allclose failed at test_t_to_ir")
        
        
            
if __name__ == '__main__':
    unittest.main()                