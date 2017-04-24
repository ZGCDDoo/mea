#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os
import unittest

from mea.model import nambu

currentdir = os.path.join(os.getcwd(), "mea/tests")

class TestNambu(unittest.TestCase):
    """ A class that implements tests for the Auxiliary green function class. If input is given 
        as a gf file of a cluster, then it builds an auxiliary GF (it can do so
        for real or complex frequencies. However, imaginary frequencies is implicit).
        However, if an auxiliary green function file is given, 
        it can extract the gflf energy (it is implicit then that the frequencies
        are on the real axis"""
    
    @classmethod
    def setUpClass(TestNambu):
        print("In test nambu.\n")

    def test_read_nambu(self):
        """ """
        fin_gf_to = os.path.join(currentdir, "files/self_short_sc_moy.dat") 
        (z_n, gf_c) = nambu.read_nambu_c(fin_gf_to, zn_col=0)
        
        z_n_test = np.array([5.235989999999981354e-02, 1.570799999999990260e-01])
        
        try:
            np.testing.assert_allclose(z_n, z_n_test , rtol=1e-7, atol=1e-7)
        except AssertionError:
            self.fail("np.testing.assert_array_allclose failed at test_read_green for z_n")        
        
        gf_t = np.array([
                         [2.278956471734890243e+00 + 1.0j*(-2.516347933723195873e-01), 1.772930116959064994e+00 + 1.0j*(-2.989305575048732400e-01),
                         -7.390005575048741449e-01 + 1.0j*9.285289005847949251e-03, 1.219453001949319271e+00 + 1.0j*1.549506920077972083e-01,
                          1.159385204678361703e+00 + 1.0j*2.237519161793372902e-01, 8.387359746588695097e-01 + 1.0j*(-2.453440000000000482e-02)],
                          [2.337720974658869189e+00 + 1.0j*(-6.926155068226119704e-01), 1.928990331384017320e+00 +1.0j*(-8.016948771929824913e-01),
                           -7.366408343079924315e-01 + 1.0j*2.657390136452241966e-02, 1.143356666666668575e+00 + 1.0j*4.120808362573097905e-01,
                            9.995260448343090687e-01 +1.0j*5.800627680311893908e-01, 7.128331949317741056e-01 +1.0j*(-5.551320097465884534e-02)]
                            ])
                                  
        gf_c_test = nambu.t_to_c(z_n, gf_t)
        gf_t_test = nambu.c_to_t(z_n, gf_c_test)
      
        try:
            np.testing.assert_allclose(gf_c, gf_c_test, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_t, gf_t_test, rtol=1e-7, atol=1e-7)
        except AssertionError:
            self.fail("np.testing.assert_allclose failed at test_read_green for gf_t") 

        # now test for gf_c
        gf_c_test = np.zeros((z_n.shape[0], 8, 8), dtype=complex)
        #gf_normal_up
        gf_c_test[:, 0, 0] = gf_t[:, 0] ; gf_c_test[:, 0, 1] = gf_t[:, 2] ; gf_c_test[:, 0, 2] = gf_t[:, 3] ; gf_c_test[:, 0, 3] = gf_t[:, 2]
        gf_c_test[:, 1, 0] = gf_c_test[:, 0, 1]; gf_c_test[:, 1, 1] = gf_t[:, 1] ; gf_c_test[:, 1, 2] = gf_c_test[:, 0, 1] ; gf_c_test[:, 1, 3] = gf_t[:, 4]
        gf_c_test[:, 2, 0] = gf_c_test[:, 0, 2] ; gf_c_test[:, 2, 1] = gf_c_test[:, 1, 2] ; gf_c_test[:, 2, 2] = gf_c_test[:, 0, 0] ; gf_c_test[:, 2, 3] = gf_c_test[:, 0, 1]
        gf_c_test[:, 3, 0] = gf_c_test[:, 0, 3] ; gf_c_test[:, 3, 1] = gf_c_test[:, 1, 3] ; gf_c_test[:, 3, 2] = gf_c_test[:, 2, 3] ; gf_c_test[:, 3, 3] = gf_c_test[:, 1, 1]
        
        #-gf_normal_down(-tau)
        gf_c_test[:, 4::, 4::] = -np.conjugate(gf_c_test[:, :4:, :4:].copy())
        
        #Nambu 
        gf_c_test[:, 0, 5] = gf_t[:, -1] ; gf_c_test[:, 0, 7] = -gf_t[:, -1]
        gf_c_test[:, 1, 4] = np.conjugate(gf_t[:, -1]) ; gf_c_test[:, 1, 6] = -np.conjugate(gf_t[:, -1]) 
        gf_c_test[:, 2, 5] = -gf_t[:, -1] ; gf_c_test[:, 2, -1] = gf_t[:, -1]
        gf_c_test[:, 3, 4] = -np.conjugate(gf_t[:, -1]) ; gf_c_test[:, 3, 6] = np.conjugate(gf_t[:, -1])
        
        gf_c_test[:, 4::, :4:] = np.swapaxes(np.conjugate(gf_c_test[:, :4:, 4::].copy()), 1, 2)

        gf_c_test_07 = -gf_t[:, -1]
        gf_c_test_00 = [2.278956471734890243e+00 + 1.0j*(-2.516347933723195873e-01), 2.337720974658869189e+00 + 1.0j*(-6.926155068226119704e-01)]           
        gf_c_test_70 = -np.conjugate(gf_t[:, -1])


        try:
            np.testing.assert_allclose(gf_c, gf_c_test, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_c[:, 0, -1], gf_c_test_07, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_c[:, 0, 0], gf_c_test_00, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_c[:, -1, 0], gf_c_test_70, rtol=1e-7, atol=1e-7)
        except AssertionError:
            self.fail("np.testing.assert_allclose failed at test_read_nambu for gf_c")
            

    def test_read_nambu_infty(self):
        """ """
        fin_gf_to = os.path.join(currentdir, "files/self_short_sc_moy.dat") 
        (z_n, gf_c) = nambu.read_nambu_c(fin_gf_to, 0)
        gf_infty = nambu.read_nambu_infty(gf_c)
        gf_test_infty = gf_c[-1, :, :]

        try:
            np.testing.assert_allclose(gf_infty, gf_test_infty, rtol=1e-7, atol=1e-7)
        except AssertionError:
             self.fail("np all close failed at test_read_infty")            

 
    def test_t_to_c(self):
        fin_gf_to = os.path.join(currentdir,"files/self_short_sc_moy.dat")
        (z_n, gf_c) = nambu.read_nambu_c(fin_gf_to, 0)

        gf_to_test = nambu.c_to_to(z_n, gf_c)
        gf_t_test = nambu.c_to_t(z_n, gf_c)
        
        gf_test1_c = nambu.to_to_c(z_n, gf_to_test)
        gf_test2_c = nambu.t_to_c(z_n, gf_t_test)

        try:
            np.testing.assert_allclose(gf_c, gf_test1_c, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_c, gf_test2_c, rtol=1e-7, atol=1e-7)
        except AssertionError:
            self.fail("Ayaya, np.testing.assert_allclose failed at test_t_to_c")    
        

    def test_c_to_ir(self):
        """ """
        fin_gf_to = os.path.join(currentdir,"files/self_short_sc_moy.dat")
        (z_n, gf_c) = nambu.read_nambu_c(fin_gf_to, 0)

        # test to see if going from c_to_ir and back gives the same thing
        gf_ir = nambu.c_to_ir(gf_c)
        gf_c_test = nambu.ir_to_c(gf_ir)
        
        gf_ir_test = np.zeros(gf_c.shape, dtype=complex)
        gf_ir_test[:, 0, 0] = gf_c[:, 0, 0] + gf_c[:, 0 ,2]
        gf_ir_test[:, 0, 1] = 2.0*gf_c[:, 0, 1]
        gf_ir_test[:, 1, 1] = gf_c[:, 1, 1] + gf_c[:, 1, 3]
        gf_ir_test[:, 2, 2] = gf_c[:, 0, 0] - gf_c[:, 0, 2]
        gf_ir_test[:, 3, 3] = gf_c[:, 1, 1] - gf_c[:, 1, 3]
        
        #print(gf_ir.shape)
        #print(gf_ir[0, 0, 3])
        zeros = np.zeros((gf_ir.shape[0]))
        try:
            #pass
            np.testing.assert_allclose(gf_c, gf_c_test, rtol=1e-7, atol=1e-7)
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

           
            np.testing.assert_allclose(gf_ir[:, 2, -1], 2.0*gf_c[:, 0, 5], rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir[:, 3, -2], np.conjugate(2.0*gf_c[:, 0, 5]), rtol=1e-7, atol=1e-7)
        except AssertionError:
            self.fail("Ayaya, np.testing.assert_allclose failed at test_t_to_c")  
    

    def test_t_to_ir(self):
        """test t_to_ir, to_to_ir and ir_to_to"""
    
        fin_gf_to = os.path.join(currentdir,"files/self_short_sc_moy.dat")
        (z_n, gf_c) = nambu.read_nambu_c(fin_gf_to, 0)

        # test to see if going from c_to_ir and back gives the same thing
        gf_ir = nambu.c_to_ir(gf_c)    
        gf_t = nambu.ir_to_t(z_n, gf_ir)
        gf_to = nambu.ir_to_to(z_n, gf_ir)
        gf_ir_test = nambu.to_to_ir(z_n, gf_to)
        gf_t_test = nambu.to_to_t(gf_to)
        gf_ir_test2 = nambu.t_to_ir(z_n, gf_t)
        
        try:
            #pass
            #print(gf_ir - gf_ir_test)
            np.testing.assert_allclose(gf_ir, gf_ir_test, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_ir, gf_ir_test2, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(gf_t, gf_t_test, rtol=1e-7, atol=1e-7)
        except AssertionError:
            self.fail("Ayaya, np.testing.assert_allclose failed at test_t_to_ir")
        
        
            
if __name__ == '__main__':
    unittest.main()                