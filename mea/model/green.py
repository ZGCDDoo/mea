#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Here every gf is a gfvec, i.e every green-function is a vector of green function.
To lighten the writing we supress the "vec". Maybe consider putting everything as in the README.txt.
"""

import numpy as np



def read_green_c(fin_gf_to, zn_col=0):
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
    gf_c = to_to_c(zn_vec, gf_to) 
    return (zn_vec, gf_c)


def read_green_ir(fin_gf_to, zn_col=0):
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
    gf_ir = to_to_ir(zn_vec, gf_to) 
    return (zn_vec, gf_ir)

def read_green_infty(gf):
    """read the high frequency part of the self-energy of the cluster"""

    assert(gf.shape[1] == gf.shape[2]) ; # assert(gf.shape[1] == 4)
    return gf[-1, :, :].copy()
    
    
def to_to_t(gf_to):
    gf_to = gf_to.copy()
    return (1.0*gf_to[:, 1::2] + 1.0j*gf_to[:, 2::2])


def t_to_to(zn_vec, gf_t):
    """ """
    zn_vec = zn_vec.copy() ; gf_t = gf_t.copy()    
    assert(zn_vec.shape[0] == gf_t.shape[0])
    assert(gf_t.shape[1] == 5)
    
    gf_to = np.zeros((gf_t.shape[0], 2*gf_t.shape[1] + 1))

    gf_to[:, 0] = zn_vec
    gf_to[:, 1::2] =  gf_t.real
    gf_to[:, 2::2] =  gf_t.imag    
    
    return gf_to


def t_to_c(zn_vec, gf_t):
    """convert from tabular form to cluster form """
    zn_vec = zn_vec.copy() ; gf_t = gf_t.copy()
    assert(zn_vec.shape[0] == gf_t.shape[0])
    assert(gf_t.shape[1] == 5)
    
    gf_c = np.zeros((zn_vec.shape[0], 4, 4), dtype=complex)

    gf_c[:, 0, 0] = gf_t[:, 0] ; gf_c[:, 0, 1] = gf_t[:, 2] ; gf_c[:, 0, 2] = gf_t[:, 3] ; gf_c[:, 0, 3] = gf_t[:, 2]
    gf_c[:, 1, 0] = gf_t[:, 2] ; gf_c[:, 1, 1] = gf_t[:, 1] ; gf_c[:, 1, 2] = gf_t[:, 2] ; gf_c[:, 1, 3] = gf_t[:, 4]
    gf_c[:, 2, 0] = gf_t[:, 3] ; gf_c[:, 2, 1] = gf_t[:, 2] ; gf_c[:, 2, 2] = gf_t[:, 0] ; gf_c[:, 2, 3] = gf_t[:, 2]
    gf_c[:, 3, 0] = gf_t[:, 2] ; gf_c[:, 3, 1] = gf_t[:, 4] ; gf_c[:, 3, 2] = gf_t[:, 2] ; gf_c[:, 3, 3] = gf_t[:, 1]

    return gf_c
    
    
def c_to_t(zn_vec, gf_c):
    """ """
    zn_vec = zn_vec.copy() ; gf_c = gf_c.copy()
    gf_to = c_to_to(zn_vec, gf_c)
    gf_t = to_to_t(gf_to)
    return gf_t
    
    
def to_to_c(zn_vec, gf_to):
    """convert from tabular-out form to cluster form"""
    zn_vec = zn_vec.copy() ; gf_to = gf_to.copy()
    assert(zn_vec.shape[0] == gf_to.shape[0])
    assert(gf_to.shape[1] == 11)
    
    #the following three lines could be replaced by: gf_t = to_to_t(gf_to)
    # but I do this for tests purposes (np.testing)
    gf_t = np.zeros((gf_to.shape[0], (gf_to.shape[1] -1)//2))
    gf_t = 1.0*gf_to[:, 1::2] + 1.0j*gf_to[:, 2::2]
    np.testing.assert_allclose(gf_t, to_to_t(gf_to))

    gf_c = np.zeros((zn_vec.shape[0], 4, 4), dtype=complex)

    gf_c[:, 0, 0] = gf_t[:, 0] ; gf_c[:, 0, 1] = gf_t[:, 2] ; gf_c[:, 0, 2] = gf_t[:, 3] ; gf_c[:, 0, 3] = gf_t[:, 2]
    gf_c[:, 1, 0] = gf_t[:, 2] ; gf_c[:, 1, 1] = gf_t[:, 1] ; gf_c[:, 1, 2] = gf_t[:, 2] ; gf_c[:, 1, 3] = gf_t[:, 4]
    gf_c[:, 2, 0] = gf_t[:, 3] ; gf_c[:, 2, 1] = gf_t[:, 2] ; gf_c[:, 2, 2] = gf_t[:, 0] ; gf_c[:, 2, 3] = gf_t[:, 2]
    gf_c[:, 3, 0] = gf_t[:, 2] ; gf_c[:, 3, 1] = gf_t[:, 4] ; gf_c[:, 3, 2] = gf_t[:, 2] ; gf_c[:, 3, 3] = gf_t[:, 1]
    
    return gf_c

def c_to_to(zn_vec, gf_c):
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


def c_to_ir(gf_c):
    """ """
    gf_c = gf_c.copy()
    U = 1/np.sqrt(2)*np.array([
                 [1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0],
                 [1.0, 0.0, -1.0, 0.0],
                 [0.0, 1.0, 0.0, -1.0] 
                 ])
                 
    Udag = np.transpose(np.conjugate(U))
    gf_ir = np.zeros(gf_c.shape, dtype=complex) 
    
    for (i, gf) in enumerate(gf_c):
        gf_ir[i] = np.dot(U, np.dot(gf, Udag))
    return gf_ir


def ir_to_c(gf_ir):
    """ """
    gf_ir = gf_ir.copy()
    U = 1/np.sqrt(2)*np.array([
                 [1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0],
                 [1.0, 0.0, -1.0, 0.0],
                 [0.0, 1.0, 0.0, -1.0] 
                 ])
    Udag = np.transpose(np.conjugate(U))    
    gf_c = np.zeros(gf_ir.shape, dtype=complex)
    
    for (i, gf) in enumerate(gf_ir):             
        gf_c[i] = np.dot(Udag, np.dot(gf, U))
    return gf_c


def ir_to_to(zn_vec, gf_ir):
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
    
    
def to_to_ir(zn_vec, gf_to):
    """convert from tabular-out form to cluster form"""
    zn_vec = zn_vec.copy() ; gf_to = gf_to.copy()
    assert(zn_vec.shape[0] == gf_to.shape[0])
    assert(gf_to.shape[1] == 11)
    
    #the following three lines could be replaced by: gf_t = to_to_t(gf_to)
    # but I do this for tests purposes (np.testing)    
    gf_t = np.zeros((gf_to.shape[0], (gf_to.shape[1] -1)//2))
    gf_t = 1.0*gf_to[:, 1::2] + 1.0j*gf_to[:, 2::2]
    np.testing.assert_allclose(gf_t, to_to_t(gf_to))

    gf_ir = np.zeros((zn_vec.shape[0], 4, 4), dtype=complex)
    gf_ir[:, 0, 0] = gf_t[:, 0] ; gf_ir[:, 0, 1] = gf_t[:, 4] 
    gf_ir[:, 1, 0] = gf_ir[:, 0, 1] ; gf_ir[:, 1, 1] = gf_t[:, 1] 
    gf_ir[:, 2, 2] = gf_t[:, 2] ;  gf_ir[:, 3, 3] = gf_t[:, 3]
    
    return gf_ir


def ir_to_t(zn_vec, gf_ir):
    """ """
    zn_vec = zn_vec.copy() ; gf_ir = gf_ir.copy()
    gf_to = ir_to_to(zn_vec, gf_ir)
    gf_t = gf_to[:, 1::2] + 1.0j*gf_to[:, 2::2]
    np.testing.assert_allclose(gf_t, to_to_t(gf_to))
    return gf_t



def t_to_ir(zn_vec, gf_t):
    """convert from tabular form to irreducible form """
    zn_vec = zn_vec.copy() ; gf_t = gf_t.copy()
    assert(zn_vec.shape[0] == gf_t.shape[0])
    assert(gf_t.shape[1] == 5)

    gf_ir = np.zeros((zn_vec.shape[0], 4, 4), dtype=complex)

    gf_ir[:, 0, 0] = gf_t[:, 0] ; gf_ir[:, 0, 1] = gf_t[:, 4] 
    gf_ir[:, 1, 0] = gf_t[:, 4] ; gf_ir[:, 1, 1] = gf_t[:, 1] ; 
    gf_ir[:, 2, 2] = gf_t[:, 2] ; gf_ir[:, 3, 3] = gf_t[:, 3]

    return gf_ir
    

def save_gf_ir(fout, zn_vec, gf_ir):
    """takes a gf in matrix form and saves it in tabular form"""
    gf_to = ir_to_to(zn_vec, gf_ir)
    np.savetxt(fout, gf_to)


def save_gf_c(fout, zn_vec, gf_c):
    """ """
    gf_to = c_to_to(zn_vec, gf_c)
    np.savetxt(fout, gf_to)