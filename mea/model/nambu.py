#!/usr/bin/env python
# -*- coding: utf-8 -*-
#creating new branch MaxEntAux
"""

"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)

import numpy as np
from . import green

def read_nambu_c(fin_nambu_to, zn_col=0):
    """Reads the cluster nambu in a tabular-out form in the cluster
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
    nambu_to = np.loadtxt(fin_nambu_to)
    
    # Get the Matsubara frequencies on imaginary axis
    z_n = nambu_to[:, zn_col].copy()

    # construct complex nambu_t (each nambu is formed by the consecutif real part 
    # (pair columns) and imaginary part (odd columns))
    nambu_c = to_to_c(z_n, nambu_to) 
    return (z_n, nambu_c)

# same as in green.py so maybe do a sort of virutal fonction or simply call green to avoid code duplication
def read_nambu_infty(nambu):
    return green.read_green_infty(nambu)
    

# same as in green.py so maybe do a sort of virutal fonction or simply call green to avoid code duplication
def to_to_t(nambu_to):
    return green.to_to_t(nambu_to)


# same as in green.py so maybe do a sort of virutal fonction or simply call green to avoid code duplication
def t_to_to(z_n, nambu_t):
    """ """
    z_n = z_n.copy() ; nambu_t = nambu_t.copy()    
    assert(z_n.shape[0] == nambu_t.shape[0])
    assert(nambu_t.shape[1] == 6)
    
    nambu_to = np.zeros((nambu_t.shape[0], 2*nambu_t.shape[1] + 1))

    nambu_to[:, 0] = z_n
    nambu_to[:, 1::2] =  nambu_t.real
    nambu_to[:, 2::2] =  nambu_t.imag    
    
    return nambu_to


def t_to_c(z_n, nambu_t):
    """convert from tabular form to cluster form """
    z_n = z_n.copy() ; nambu_t = nambu_t.copy()
    assert(z_n.shape[0] == nambu_t.shape[0])
    assert(nambu_t.shape[1] == 6)
    
    gf_normal_up = np.zeros((z_n.shape[0], 4, 4), dtype=complex)
    gf_normal_down = np.zeros((z_n.shape[0], 4, 4), dtype=complex)
    gf_gorkov = np.zeros((z_n.shape[0], 4, 4), dtype=complex)

    gf_normal_up = green.t_to_c(z_n, nambu_t[:, :-1])
    gf_normal_down = np.conjugate(gf_normal_up.copy())

    nambu_t[:, -1] = nambu_t[:, -1].real.copy() #for d-wave the anomalous self-energy is real, thus discard the imaginary part
    gf_gorkov[:, 0, 1] = nambu_t[:, -1] ; gf_gorkov[:, 0, 3] = -nambu_t[:, -1] 
    gf_gorkov[:, 1, 0] = np.conjugate(nambu_t[:, -1]) ; gf_gorkov[:, 1, 2] = -np.conjugate(nambu_t[:, -1])
    gf_gorkov[:, 2, 1] = -nambu_t[:, -1] ; gf_gorkov[:, 2 , 3] = nambu_t[:, -1]
    gf_gorkov[:, 3, 0] = -np.conjugate(nambu_t[:, -1]) ; gf_gorkov[:, 3, 2] = np.conjugate(nambu_t[:, -1]) 

    tmp1 = np.concatenate((gf_normal_up, np.conjugate(np.swapaxes(gf_gorkov, 1, 2)) ), axis=1)
    tmp2 = np.concatenate((gf_gorkov, -gf_normal_down), axis=1)
    nambu_c = np.concatenate((tmp1, tmp2), axis=2)

    return nambu_c.copy()
    
    
    
    
def to_to_c(z_n, nambu_to):
    """convert from tabular-out form to cluster form"""
    z_n = z_n.copy() ; nambu_to = nambu_to.copy()
    assert(z_n.shape[0] == nambu_to.shape[0])
    assert(nambu_to.shape[1] == 13)
    
    #the following three lines could be replaced by: nambu_t = to_to_t(nambu_to)
    # but I do this for tests purposes (np.testing)
    nambu_t = np.zeros((nambu_to.shape[0], (nambu_to.shape[1] -1)//2))
    nambu_t = 1.0*nambu_to[:, 1::2] + 1.0j*nambu_to[:, 2::2]
    np.testing.assert_allclose(nambu_t, to_to_t(nambu_to))

    gf_normal_up = np.zeros((z_n.shape[0], 4, 4), dtype=complex)
    gf_gorkov = np.copy(gf_normal_up)

    gf_normal_up[:, 0, 0] = nambu_t[:, 0] ; gf_normal_up[:, 0, 1] = nambu_t[:, 2] ; gf_normal_up[:, 0, 2] = nambu_t[:, 3] ; gf_normal_up[:, 0, 3] = nambu_t[:, 2]
    gf_normal_up[:, 1, 0] = nambu_t[:, 2] ; gf_normal_up[:, 1, 1] = nambu_t[:, 1] ; gf_normal_up[:, 1, 2] = nambu_t[:, 2] ; gf_normal_up[:, 1, 3] = nambu_t[:, 4]
    gf_normal_up[:, 2, 0] = nambu_t[:, 3] ; gf_normal_up[:, 2, 1] = nambu_t[:, 2] ; gf_normal_up[:, 2, 2] = nambu_t[:, 0] ; gf_normal_up[:, 2, 3] = nambu_t[:, 2]
    gf_normal_up[:, 3, 0] = nambu_t[:, 2] ; gf_normal_up[:, 3, 1] = nambu_t[:, 4] ; gf_normal_up[:, 3, 2] = nambu_t[:, 2] ; gf_normal_up[:, 3, 3] = nambu_t[:, 1]
    
    gf_normal_down = np.conjugate(gf_normal_up)

    nambu_t[:, -1] = nambu_t[:, -1].real.copy() #for d-wave the anomalous self-energy is real, thus discard the imaginary part
    gf_gorkov[:, 0, 1] = nambu_t[:, -1] ; gf_gorkov[:, 0, 3] = -nambu_t[:, -1]
    gf_gorkov[:, 1, 0] = np.conjugate(nambu_t[:, -1]) ; gf_gorkov[:, 1, 2] = -np.conjugate(nambu_t[:, -1])
    gf_gorkov[:, 2, 1 ] = -nambu_t[:, -1] ; gf_gorkov[:, 2, 3] = nambu_t[:, -1]
    gf_gorkov[:, 3, 0] = -np.conjugate(nambu_t[:, -1]) ; gf_gorkov[:, 3, 2] = np.conjugate(nambu_t[:, -1])

    tmp1 = np.concatenate((gf_normal_up, np.swapaxes(np.conjugate(gf_gorkov), 1, 2)), axis=1)
    tmp2 = np.concatenate((gf_gorkov, -gf_normal_down), axis=1)
    nambu_c = np.concatenate((tmp1, tmp2), axis=2)

    return nambu_c

def c_to_to(z_n, nambu_c):
    """convert from cluster form to tabular-out form """
    z_n = z_n.copy() ; nambu_c = nambu_c.copy()
    assert(z_n.shape[0] == nambu_c.shape[0])
    assert(nambu_c.shape[1] == nambu_c.shape[2]) ; assert(nambu_c.shape[1] == 8)

    nambu_to = np.zeros((z_n.shape[0], 13))
    
    nambu_to[:, 0] = z_n
    nambu_to[:, 1] = nambu_c[:, 0, 0].real ; nambu_to[:, 2] = nambu_c[:, 0, 0].imag
    nambu_to[:, 3] = nambu_c[:, 1, 1].real ; nambu_to[:, 4] = nambu_c[:, 1, 1].imag
    nambu_to[:, 5] = nambu_c[:, 0, 1].real ; nambu_to[:, 6] = nambu_c[:, 0, 1].imag
    nambu_to[:, 7] = nambu_c[:, 0, 2].real ; nambu_to[:, 8] = nambu_c[:, 0, 2].imag
    nambu_to[:, 9] = nambu_c[:, 1, 3].real ; nambu_to[:, 10] = nambu_c[:, 1, 3].imag  
    nambu_to[:, 11] = nambu_c[:, 0, 5].real ; nambu_to[:, 12] = nambu_c[:, 0, 5].imag

    return nambu_to     


def c_to_t(z_n, nambu_c):
    """ """
    z_n = z_n.copy() ; nambu_c = nambu_c.copy()
    nambu_to = c_to_to(z_n, nambu_c)
    nambu_t = to_to_t(nambu_to)
    return nambu_t


def c_to_ir(nambu_c):
    """ """
    nambu_c = nambu_c.copy()
    U = 1/np.sqrt(2)*np.array([
                 [1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0],
                 [1.0, 0.0, -1.0, 0.0],
                 [0.0, 1.0, 0.0, -1.0] 
                 ])

                 
    Udag = np.transpose(np.conjugate(U))
    U_nambu = np.kron(np.eye(2), U)
    Udag_nambu = np.kron(np.eye(2), Udag)
    nambu_ir = np.zeros(nambu_c.shape, dtype=complex) 
    
    for (i, nambu) in enumerate(nambu_c):
        nambu_ir[i] = np.dot(U_nambu, np.dot(nambu, Udag_nambu))
    return nambu_ir


def ir_to_c(nambu_ir):
    """ """
    nambu_ir = nambu_ir.copy()
    U = 1/np.sqrt(2)*np.array([
                 [1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0],
                 [1.0, 0.0, -1.0, 0.0],
                 [0.0, 1.0, 0.0, -1.0] 
                 ])
    Udag = np.transpose(np.conjugate(U))    
    nambu_c = np.zeros(nambu_ir.shape, dtype=complex)
    U_nambu = np.kron(np.eye(2), U)
    Udag_nambu = np.kron(np.eye(2), Udag)
    for (i, nambu) in enumerate(nambu_ir):             
        nambu_c[i] = np.dot(Udag_nambu, np.dot(nambu, U_nambu))
    return nambu_c


def ir_to_to(z_n, nambu_ir):
    """convert from irreducible form to tabular-out form """
    z_n = z_n.copy() ; nambu_ir = nambu_ir.copy()
    assert(z_n.shape[0] == nambu_ir.shape[0])
    assert(nambu_ir.shape[1] == nambu_ir.shape[2]) ; assert(nambu_ir.shape[1] == 8)

    nambu_to = np.zeros((z_n.shape[0], 13))
    
    nambu_to[:, 0] = z_n
    nambu_to[:, 1] = nambu_ir[:, 0, 0].real ; nambu_to[:, 2] = nambu_ir[:, 0, 0].imag
    nambu_to[:, 3] = nambu_ir[:, 1, 1].real ; nambu_to[:, 4] = nambu_ir[:, 1, 1].imag
    nambu_to[:, 5] = nambu_ir[:, 2, 2].real ; nambu_to[:, 6] = nambu_ir[:, 2, 2].imag
    nambu_to[:, 7] = nambu_ir[:, 3, 3].real ; nambu_to[:, 8] = nambu_ir[:, 3, 3].imag
    nambu_to[:, 9] = nambu_ir[:, 0, 1].real ; nambu_to[:, 10] = nambu_ir[:, 0, 1].imag   
    nambu_to[:, 11] = nambu_ir[:, 2, -1].real; nambu_to[:, 12] = nambu_ir[:, 2, -1].imag
    return nambu_to 
    
    
def to_to_ir(z_n, nambu_to):
    """convert from tabular-out form to cluster form"""
    z_n = z_n.copy() ; nambu_to = nambu_to.copy()
    assert(z_n.shape[0] == nambu_to.shape[0])
    assert(nambu_to.shape[1] == 13)
    
    #the following three lines could be replaced by: nambu_t = to_to_t(nambu_to)
    # but I do this for tests purposes (np.testing)    
    nambu_t = np.zeros((nambu_to.shape[0], (nambu_to.shape[1] -1)//2))
    nambu_t = 1.0*nambu_to[:, 1::2] + 1.0j*nambu_to[:, 2::2]
    np.testing.assert_allclose(nambu_t, to_to_t(nambu_to))

    gf_normal_up = np.zeros((z_n.shape[0], 4, 4), dtype=complex)
    gf_normal_down = np.copy(gf_normal_up)
    gf_gorkov = np.copy(gf_normal_up)

    gf_normal_up[:, 0, 0] = nambu_t[:, 0] ; gf_normal_up[:, 0, 1] = nambu_t[:, 4] 
    gf_normal_up[:, 1, 0] = gf_normal_up[:, 0, 1] ; gf_normal_up[:, 1, 1] = nambu_t[:, 1] 
    gf_normal_up[:, 2, 2] = nambu_t[:, 2] ;  gf_normal_up[:, 3, 3] = nambu_t[:, 3]

    gf_normal_down = np.conjugate(gf_normal_up)

    nambu_t[:, -1] = nambu_t[:, -1].real.copy() #for d-wave the anomalous self-energy is real, thus discard the imaginary part
    gf_gorkov[:, 2, 3] = nambu_t[:, -1]
    gf_gorkov[:, 3, 2] = np.conjugate(nambu_t[:, -1])

    tmp1 = np.concatenate((gf_normal_up, np.swapaxes(np.conjugate(gf_gorkov), 1, 2)), axis=1)
    tmp2 = np.concatenate((gf_gorkov, -gf_normal_down), axis=1)
    nambu_ir = np.concatenate((tmp1, tmp2), axis=2)

    return nambu_ir


def ir_to_t(z_n, nambu_ir):
    """ """
    z_n = z_n.copy() ; nambu_ir = nambu_ir.copy()
    nambu_to = ir_to_to(z_n, nambu_ir)
    nambu_t = nambu_to[:, 1::2] + 1.0j*nambu_to[:, 2::2]
    np.testing.assert_allclose(nambu_t, to_to_t(nambu_to))
    return nambu_t

def t_to_ir(z_n, nambu_t):
    """ """
    z_n = z_n.copy();  nambu_t = nambu_t.copy()
    assert(z_n.shape[0] == nambu_t.shape[0])
    gf_normal_up = green.t_to_ir(z_n, nambu_t[:, :-1:])
    gf_normal_down = np.conjugate(gf_normal_up)

    gf_gorkov = np.zeros((z_n.shape[0], 4, 4), dtype=complex)
    nambu_t[:, -1] = nambu_t[:, -1].real.copy() #for d-wave the anomalous self-energy is real, thus discard the imaginary part
    gf_gorkov[:, 2, 3] = nambu_t[:, -1]
    gf_gorkov[:, 3, 2] = np.conjugate(nambu_t[:, -1])

    tmp1 = np.concatenate((gf_normal_up, np.swapaxes(np.conjugate(gf_gorkov), 1, 2)), axis=1)
    tmp2 = np.concatenate((gf_gorkov, -gf_normal_down), axis=1)
    nambu_ir = np.concatenate((tmp1, tmp2), axis=2)

    return nambu_ir


def get_normalup_ir(nambu_ir):
    """ """
    return (nambu_ir[:, :4:, :4:].copy())


def get_anormal_ir(nambu_ir):
    """ """
    return (nambu_ir[:, 4::, 4::].copy())
    

def save_nambu_ir(fout, z_n, nambu_ir):
    """takes a nambu in matrix form and saves it in tabular form"""
    nambu_to = ir_to_to(z_n, nambu_ir)
    np.savetxt(fout, nambu_to)


def save_nambu_c(fout, z_n, nambu_c):
    """ """
    nambu_to = c_to_to(z_n, nambu_c)
    np.savetxt(fout, nambu_to)