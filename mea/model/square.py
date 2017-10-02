import numpy as np  # type: ignore
from scipy import linalg # type: ignore
from scipy.integrate import dblquad # type: ignore
from typing import Tuple


#Equivalent of AnisotropicTriangularPlaquette
from .import abc_model

class Square(abc_model.ABCModel):
    """ """
    def __init__(self, t: float, tp: float, tpp: float, mu: float, z_vec, sEvec_c) -> None:
        """ """
        super().__init__(t, tp, tpp, mu, z_vec, sEvec_c)
        return None



    def t_value(self, kx: float, ky: float) : # This is t_ij(k_tilde)
        """this t_value is only good if tpp = 0.0"""
        t = self.t ; tp = self.tp; tpp = self.tpp
        t_val = np.zeros((4, 4), dtype=complex)
        ex = np.exp(-2.0j*kx) ; emx = np.conjugate(ex)
        ey = np.exp(-2.0j*ky) ; emy = np.conjugate(ey)
        tloc = np.array([[0.0, -t, -tp, -t],
                            [-t, 0.0, -t, 0.0],
                            [-tp, -t, 0.0, -t],
                            [-t, 0.0, -t, 0.0]])


        t_val += tloc

        t_val[0, 0] += -tpp*(ex+emx+ey+emy);      t_val[0, 1] += -t*ex;                   t_val[0, 2] += -tp*(ex + ey + ex*ey);    t_val[0, 3] += -t*ey
        t_val[1, 0] += -t*emx;                    t_val[1, 1] += -tpp*(ex+emx+ey+emy);    t_val[1, 2] += -t*ey;                    t_val[1, 3] += -tp*(emx + ey + emx*ey)
        t_val[2, 0] += -tp*(emx + emy + emx*emy); t_val[2, 1] += -t*emy;                  t_val[2, 2] += -tpp*(ex+emx+ey+emy);     t_val[2, 3] += -t*emx
        t_val[3, 0] += -t*emy;                    t_val[3, 1] += -tp*(ex + emy + ex*emy); t_val[3, 2] += -t*ex;                    t_val[3, 3] += -tpp*(ex+emx+ey+emy)

        return (t_val)



    
    def eps_0(self, kx: float, ky: float) -> float:
        """the free dispersion relation for the given Model"""

        t: float = self.t ; tp: float = self.tp; tpp = self.tpp

        return (2.0*(-t*(np.cos(kx) + np.cos(ky)) -  tp*(np.cos(kx + ky) + np.cos(kx-ky)) - tpp*(np.cos(2.0*kx) + np.cos(2.0*ky))) ) 
                        




    
    def hopping_test(self, kx: float, ky: float):
        """A different approach to calculate t_value (t(ktilde)) using the explicit fourier transform """

        k = np.array([kx, ky])
        N_c: int = self._Nc
        r_sites = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        K_sites = np.array([[0.0, 0.0], [np.pi, 0.0], [np.pi, np.pi], [0.0, np.pi]])
        t_array = np.zeros((N_c, N_c), dtype=complex)

        for i in range(N_c):
            for j in range(N_c):
                for K in K_sites:
                    t_array[i, j] += 1/N_c * np.exp(1.0j*np.dot(K + k, r_sites[i] - r_sites[j])) * self.eps_0(*(K + k))

        return t_array