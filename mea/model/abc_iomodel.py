#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Here every gf is a gfvec, i.e every green-function is a vector of green function.
To lighten the writing we supress the "vec". Maybe consider putting everything as in the README.txt.
"""

import numpy as np
import abc


class ABCIOModel(abc.ABC):


    def __init__(self) -> None:
        return None

    @property
    @abc.abstractmethod
    def U(self) -> None:
        return None


    @abc.abstractmethod
    def to_to_c(self, zn_vec, gf_to) -> None:
        """convert from tabular-out form to cluster form"""
        return None


    @abc.abstractmethod
    def c_to_to(self, zn_vec, gf_c) -> None:
        """convert from cluster form to tabular-out form """
        return None




    @abc.abstractmethod
    def ir_to_to(self, zn_vec, gf_ir) -> None:
        """convert from irreducible form to tabular-out form """
        return None


    @abc.abstractmethod
    def to_to_ir(self, zn_vec, gf_to) -> None:
        """convert from tabular-out form to cluster form"""
        return None


    def read_green_c(self, fin_gf_to, zn_col=0):
        """Reads the cluster green in a tabular-out form in the cluster
        indices and returns the matsubara grid (first return paramter) 
        and the cluster-matrix form (second return parameter)

        Args:

        Keywords Args:
            zn_col (None||int): the col  which holds the matsubara frequencies
            fin_sE_to (str): the file which holds the sE_to file from Patrick's 
                Program 
            
        Returns: None

        Raises:
        """
        gf_to = np.loadtxt(fin_gf_to)
        
        # Get the Matsubara frequencies on imaginary axis
        zn_vec = gf_to[:, zn_col].copy()

        # construct complex gf_t (each gf is formed by the consecutif real part 
        # (pair columns) and imaginary part (odd columns))
        gf_c = self.to_to_c(zn_vec, gf_to) 
        return (zn_vec, gf_c)


    def read_green_ir(self, fin_gf_to, zn_col=0):
        """Reads the irreducible green in a tabular-out form in the cluster
        indices and returns the matsubara grid (first return paramter) 
        and the cluster-matrix form (second return parameter)

        Args:

        Keywords Args:
            zn_col (None||int): the col  which holds the matsubara frequencies
            fin_sE_to (str): the file which holds the sE_to file from Patrick's 
                Program 
            
        Returns: None

        Raises:
        """
        gf_to = np.loadtxt(fin_gf_to)
        
        # Get the Matsubara frequencies on imaginary axis
        zn_vec = gf_to[:, zn_col].copy()

        # construct complex gf_t (each gf is formed by the consecutif real part 
        # (pair columns) and imaginary part (odd columns))
        gf_ir = self.to_to_ir(zn_vec, gf_to) 
        return (zn_vec, gf_ir)


    def read_green_infty(self, gf):
        """read the high frequency part of the self-energy of the cluster"""

        assert(gf.shape[1] == gf.shape[2]) ; # assert(gf.shape[1] == 4)
        return gf[-1, :, :].copy()
    


    def to_to_t(self,gf_to):
        gf_to = gf_to.copy()
        return (1.0*gf_to[:, 1::2] + 1.0j*gf_to[:, 2::2])


    def t_to_to(self, zn_vec, gf_t):
        """ """
        zn_vec = zn_vec.copy() ; gf_t = gf_t.copy()    
        assert(zn_vec.shape[0] == gf_t.shape[0])
        assert(gf_t.shape[1] == 5)
        
        gf_to = np.zeros((gf_t.shape[0], 2*gf_t.shape[1] + 1))

        gf_to[:, 0] = zn_vec
        gf_to[:, 1::2] =  gf_t.real
        gf_to[:, 2::2] =  gf_t.imag    
        
        return gf_to


    def t_to_c(self, zn_vec, gf_t) -> None:
        """convert from tabular form to cluster form """
        gf_to = self.t_to_to(zn_vec, gf_t)
        return self.to_to_c(zn_vec, gf_to)

    
    def c_to_t(self, zn_vec, gf_c):
        """ """
        gf_to = self.c_to_to(zn_vec, gf_c)
        gf_t = self.to_to_t(gf_to)
        return gf_t
    

    def c_to_ir(self, gf_c):
        """ """
        gf_c = gf_c.copy()
        Udag = np.transpose(np.conjugate(self.U))
        gf_ir = np.zeros(gf_c.shape, dtype=complex) 
        
        for (i, gf) in enumerate(gf_c):
            gf_ir[i] = np.dot(self.U, np.dot(gf, Udag))
        return gf_ir


    def ir_to_c(self, gf_ir):
        """ """
        gf_ir = gf_ir.copy()
        Udag = np.transpose(np.conjugate(self.U))    
        gf_c = np.zeros(gf_ir.shape, dtype=complex)
        
        for (i, gf) in enumerate(gf_ir):             
            gf_c[i] = np.dot(Udag, np.dot(gf, self.U))
        return gf_c


    
    def ir_to_t(self, zn_vec, gf_ir):
        """ """
        zn_vec = zn_vec.copy() ; gf_ir = gf_ir.copy()
        gf_to = self.ir_to_to(zn_vec, gf_ir)
        gf_t = gf_to[:, 1::2] + 1.0j*gf_to[:, 2::2]
        np.testing.assert_allclose(gf_t, self.to_to_t(gf_to))
        return gf_t


    def t_to_ir(self, zn_vec, gf_t) -> None:
        """convert from tabular form to irreducible form """

        gf_to = self.t_to_to(zn_vec, gf_t)
        return self.to_to_ir(zn_vec, gf_to)


    def save_gf_ir(self, fout, zn_vec, gf_ir):
        """takes a gf in matrix form and saves it in tabular form"""
        gf_to = self.ir_to_to(zn_vec, gf_ir)
        np.savetxt(fout, gf_to)


    def save_gf_c(self, fout, zn_vec, gf_c):
        """ """
        gf_to = self.c_to_to(zn_vec, gf_c)
        np.savetxt(fout, gf_to)