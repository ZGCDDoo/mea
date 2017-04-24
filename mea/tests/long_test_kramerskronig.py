#!/usr/bin/env python

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os
import unittest

from mea.tools import kramerskronig as kk 
currentdir = os.path.join(os.getcwd(), "mea/tests")

class TestKK(unittest.TestCase):
    """ A class that implements tests for the Auxiliary green function class. If input is given 
        as a sE file of a cluster, then it builds an auxiliary GF (it can do so
        for real or complex frequencies. However, imaginary frequencies is implicit).
        However, if an auxiliary green function file is given, 
        it can extract the self energy (it is implicit then that the frequencies
        are on the real axis"""
    

    def test_kk(self):
        """ """
        
        fin_kk1 = os.path.abspath(os.path.join(currentdir,"files/kk_test_z+i.txt"))
        fin_kk2 = os.path.abspath(os.path.join(currentdir,"files/kk_test_z-4.txt"))

        # delete a few of the endpoints
        fn_good1 = np.loadtxt(fin_kk1)
        fn_good2 = np.loadtxt(fin_kk2)
        
            
    
        mesh = np.linspace(-200.0, 200.0, 15000)
    
        z = (1.0 + 1.0j)*mesh
        fn1 = (z + 1.0j)**(-1.0)
        re_fn1 = fn1.real
        im_fn1 = fn1.imag
        fn2 = (z - 4.0)**(-1.0)
        re_fn2 = fn2.real
        im_fn2 = fn2.imag

        
        test_re_fn1 = kk.KramersKroning(mesh, im_fn1)
        test_re_fn2 = kk.KramersKroning(mesh, im_fn2)

        # delete a few of the endpoints
        fn1 = np.transpose([mesh, test_re_fn1, im_fn1])
        fn2 = np.transpose([mesh, test_re_fn2, im_fn2])
       
        try:
            np.testing.assert_allclose(fn1[20:-20:,:], fn_good1[20:-20:,:])
            np.testing.assert_allclose(fn2[20:-20:,:], fn_good2[20:-20:,:])
        except AssertionError:
            self.fail("ayaya np.allclose failed at test_build_gf_aux") 

if __name__ == '__main__':
    unittest.main()                   