#!/usr/bin/env python
"""
Code originnaly from Reza Nourfaken and happily shared by him.
Forked and modified by Charles-David Hebert 
"""

from types import *
import numpy as np
import warnings
from scipy.integrate import simps

def KramersKroning(mesh, imG):
    """ Obtain the real part of Fn using Kramers-Kroning relation.
        \int_a^b (dx/pi) (g(x)/(x-y)). The divergent part is removed
        by adding and substracting g(y) in numerator. At point x = y it should be calculated by estimating derivative (dg/dx)_(x=y). 
        \int_a^b (dx/pi) (g(x)/(x-y))= \int_a^b (dx/pi) ((g(x)-g(y))/(x-y)) + (g(y)/\pi)log((b-y)/(y-a))   

        args:
            mesh: the mesh on which the function is evaluated
            imG (np.ndarray(float64)): the imaginary part of a function G 
        returns:
            reG: The real part of the function G    
        raises:    
        """
    # sanity checks 
    assert(type(imG) == np.ndarray)
    assert(type(mesh) == np.ndarray)
    assert(imG.shape == mesh.shape)

    reG = np.zeros(imG.shape)
    NumMeshPoint = mesh.shape[0]

    if (abs(imG[0]) > 0.1 or abs(imG[-1]) > 0.1): 
        print("\n Im G for lower limit of integral = ", imG[0])
        print(" Im G for upper limit of integral = ", imG[-1])        
        warnings.warn(" Kramers Kronig could give wrong results")
    IntegralMesh = mesh.copy()                 # force copying     
    Integrand = np.zeros((NumMeshPoint), dtype=np.float)
    x = np.zeros((3), dtype=np.float)
    y = np.zeros((3), dtype=np.float)
    coef = np.zeros((3), dtype=np.float) 
    for n in range(NumMeshPoint):   
        for m in range(NumMeshPoint):
            if (m == n):           
                Integrand[m] = 0.0
                if ( m > 1 and m < NumMeshPoint-2):         # finding slope by fitting a second order polynominal Fn                           
                    x[0:3] = mesh[m-1:m+2].copy()           # n:m --> in python starts from n and stop at m-1 !
                    y[0:3] = imG[m-1:m+2].copy()
                    coef = np.polyfit(x, y, 2, full=False) 
                    Integrand[m] = (2.0*coef[0]*mesh[m]+coef[1])/np.pi
            else:
                Integrand[m] = (imG[m] - imG[n])/(mesh[m] - mesh[n])/(np.pi)  
        reG[n] = simps(Integrand, IntegralMesh)
#
    for n in range(2,NumMeshPoint-1):
        reG[n] += (imG[n]/(np.pi))*np.log((mesh[NumMeshPoint-1]-mesh[n])/(mesh[n]-mesh[0]))
#        
    return reG


if __name__ == "__main__":

    def Gz(x, y):
        z = x + 1.0j*y
        z_c = np.conj(z)
        #return (z + 1.0j)**(-1.0)
        return 1.0/(z - 4.0)
    
    mesh = np.linspace(-200.0, 200.0, 15000)
    #expMinusz = fz(x, x)
    gz = Gz(mesh, mesh)
    re_gz = gz.real
    im_gz = gz.imag

    test_re_gz = KramersKroning(mesh, im_gz)

    np.savetxt("gz.txt", np.array([mesh, re_gz, im_gz]).T)
    np.savetxt("gz_test.txt", np.array([mesh, test_re_gz, im_gz]).T)      