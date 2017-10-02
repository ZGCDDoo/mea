#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Here every gf is a gfvec, i.e every green-function is a vector of green function.
To lighten the writing we supress the "vec". Maybe consider putting everything as in the README.txt.
"""

import numpy as np


from .import abc_iomodel


class IOTriangle(abc_iomodel.ABCIOModel):
    """ """
    def __init__(self) -> None:
        """ """
        super().__init__()
        return None

    @property
    def U(self):
        return (1/np.sqrt(2)*np.array([
                 [1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0],
                 [1.0, 0.0, -1.0, 0.0],
                 [0.0, 1.0, 0.0, -1.0] 
                 ])
        )
    
    
    def to_to_c(self, zn_vec, gf_to):
        """convert from tabular-out form to cluster form"""
        zn_vec = zn_vec.copy() ; gf_to = gf_to.copy()
        assert(zn_vec.shape[0] == gf_to.shape[0])
        assert(gf_to.shape[1] == 11)
        
        #the following three lines could be replaced by: gf_t = to_to_t(gf_to)
        # but I do this for tests purposes (np.testing)
        gf_t = np.zeros((gf_to.shape[0], (gf_to.shape[1] -1)//2))
        gf_t = 1.0*gf_to[:, 1::2] + 1.0j*gf_to[:, 2::2]
        np.testing.assert_allclose(gf_t, self.to_to_t(gf_to))

        gf_c = np.zeros((zn_vec.shape[0], 4, 4), dtype=complex)

        gf_c[:, 0, 0] = gf_t[:, 0] ; gf_c[:, 0, 1] = gf_t[:, 2] ; gf_c[:, 0, 2] = gf_t[:, 3] ; gf_c[:, 0, 3] = gf_t[:, 2]
        gf_c[:, 1, 0] = gf_t[:, 2] ; gf_c[:, 1, 1] = gf_t[:, 1] ; gf_c[:, 1, 2] = gf_t[:, 2] ; gf_c[:, 1, 3] = gf_t[:, 4]
        gf_c[:, 2, 0] = gf_t[:, 3] ; gf_c[:, 2, 1] = gf_t[:, 2] ; gf_c[:, 2, 2] = gf_t[:, 0] ; gf_c[:, 2, 3] = gf_t[:, 2]
        gf_c[:, 3, 0] = gf_t[:, 2] ; gf_c[:, 3, 1] = gf_t[:, 4] ; gf_c[:, 3, 2] = gf_t[:, 2] ; gf_c[:, 3, 3] = gf_t[:, 1]
        
        return gf_c


    def c_to_to(self, zn_vec, gf_c):
        """convert from cluster form to tabular-out form """
        zn_vec = zn_vec.copy() ; gf_c = gf_c.copy()
        assert(zn_vec.shape[0] == gf_c.shape[0])
        assert(gf_c.shape[1] == gf_c.shape[2]) ; assert(gf_c.shape[1] == 4)

        gf_to = np.zeros((zn_vec.shape[0], 11))
        
        gf_to[:, 0] = zn_vec
        gf_to[:, 1] = gf_c[:, 0, 0].real ; gf_to[:, 2] = gf_c[:, 0, 0].imag
        gf_to[:, 3] = gf_c[:, 1, 1].real ; gf_to[:, 4] = gf_c[:, 1, 1].imag
        gf_to[:, 5] = gf_c[:, 0, 1].real ; gf_to[:, 6] = gf_c[:, 0, 1].imag
        gf_to[:, 7] = gf_c[:, 0, 2].real ; gf_to[:, 8] = gf_c[:, 0, 2].imag
        gf_to[:, 9] = gf_c[:, 1, 3].real ; gf_to[:, 10] = gf_c[:, 1, 3].imag   

        return gf_to     


    def ir_to_to(self, zn_vec, gf_ir):
        """convert from irreducible form to tabular-out form """
        zn_vec = zn_vec.copy() ; gf_ir = gf_ir.copy()
        assert(zn_vec.shape[0] == gf_ir.shape[0])
        assert(gf_ir.shape[1] == gf_ir.shape[2]) ; assert(gf_ir.shape[1] == 4)

        gf_to = np.zeros((zn_vec.shape[0], 11))
        
        gf_to[:, 0] = zn_vec
        gf_to[:, 1] = gf_ir[:, 0, 0].real ; gf_to[:, 2] = gf_ir[:, 0, 0].imag
        gf_to[:, 3] = gf_ir[:, 1, 1].real ; gf_to[:, 4] = gf_ir[:, 1, 1].imag
        gf_to[:, 5] = gf_ir[:, 2, 2].real ; gf_to[:, 6] = gf_ir[:, 2, 2].imag
        gf_to[:, 7] = gf_ir[:, 3, 3].real ; gf_to[:, 8] = gf_ir[:, 3, 3].imag
        gf_to[:, 9] = gf_ir[:, 0, 1].real ; gf_to[:, 10] = gf_ir[:, 0, 1].imag   

        return gf_to 
    
    
    def to_to_ir(self, zn_vec, gf_to):
        """convert from tabular-out form to cluster form"""
        zn_vec = zn_vec.copy() ; gf_to = gf_to.copy()
        assert(zn_vec.shape[0] == gf_to.shape[0])
        assert(gf_to.shape[1] == 11)
        
        #the following three lines could be replaced by: gf_t = to_to_t(gf_to)
        # but I do this for tests purposes (np.testing)    
        gf_t = np.zeros((gf_to.shape[0], (gf_to.shape[1] -1)//2))
        gf_t = 1.0*gf_to[:, 1::2] + 1.0j*gf_to[:, 2::2]
        np.testing.assert_allclose(gf_t, self.to_to_t(gf_to))

        gf_ir = np.zeros((zn_vec.shape[0], 4, 4), dtype=complex)
        gf_ir[:, 0, 0] = gf_t[:, 0] ; gf_ir[:, 0, 1] = gf_t[:, 4] 
        gf_ir[:, 1, 0] = gf_ir[:, 0, 1] ; gf_ir[:, 1, 1] = gf_t[:, 1] 
        gf_ir[:, 2, 2] = gf_t[:, 2] ;  gf_ir[:, 3, 3] = gf_t[:, 3]
        
        return gf_ir