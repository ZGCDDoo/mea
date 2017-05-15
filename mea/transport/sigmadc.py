import numpy as np # type: ignore
from ..model import periodize_class as perc  # type: ignore
import scipy.integrate as sI # type: ignore


class SigmaDC:

    def __init__(self, wvec, sEvec_c, beta: float, mu: float) -> None:
        """ """

        self.wvec = wvec
        self.sEvec_c = sEvec_c
        self.beta = beta
        self.mu = mu 
        self.prefactor = -2.0 #sum on spins,  weistrass-th, v_kz integrated 
        self.t = 1.0
        self.tp = 0.4
        self.cutoff = 30  # beta*omega
        self.model = perc.Model(self.t, self.tp, mu, wvec, sEvec_c)

        return None


    def dfd_dw(self, omega: float) -> float:
        """derivative of fermi-dirac  with respect to omega"""
        # implement fermi function as the exemple of the logistic function
        
        beta: float = self.beta
        result: float = -beta * np.exp(beta*omega)/(1.0 + np.exp(beta*omega))**2.0
        return result


    def y1(self, x: float) -> float:
        return -np.pi
    
    def y2(self, x: float) -> float:
        return np.pi


    def calc_sigmadc(self) -> float:
        """ """
        sigma_dc: float  = 0.0
        Akw2 = self.model.periodize_Akw2

        integrand_w = np.zeros(self.wvec.shape)
        for (i, ww) in enumerate(self.wvec):

            if abs(self.beta*ww) > self.cutoff:
                integrand_w[i] = 0.0
            else:
                integrand_w[i] = 1.0/(2.0*np.pi)**(2.0)*sI.dblquad(Akw2, -np.pi, np.pi, self.y1, self.y2, epsabs=1e-8,
                                                               args=([i]) )[0]
                integrand_w[i] *= self.dfd_dw(ww)
            
        sigma_dc = 1.0/(2.0*np.pi)*sI.simps(integrand_w, self.wvec)

        
        sigma_dc *= self.prefactor

        print("sigma_dc = ", sigma_dc)
        with open("sigmadc.dat", mode="a") as fout:
            fout.write(str(sigma_dc) + "\n")
            
        return sigma_dc


    def calc_sigmadc_test(self, Aw) -> float: 
        """ calculate sigma_dc with a non-interacting A(w)given, independant of k.
            A(k,w) = A(w)(2pi)**2.0*delta(k)
            """

        sigma_dc: float  = 0.0
        Aw2_vec = Aw[:, 1]*Aw[:, 1]  
        wwvec = Aw[:, 0]
        
        dfd_dw_vec = np.zeros(wwvec.shape)
        for (i, ww) in enumerate(wwvec):
            dfd_dw_vec[i] = self.dfd_dw(ww)

        sigma_dc = -1.0/(2.0*np.pi)*self.prefactor*sI.simps(Aw2_vec * dfd_dw_vec, wwvec)

        return sigma_dc




