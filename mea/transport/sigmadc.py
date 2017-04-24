import numpy as np # type: ignore
import periodize  # type: ignore
import scipy.integrate as sI # type: ignore
import green # type: ignore


class SigmaDC:

    def __init__(self, wvec, sEvec_c, beta: float, mu: float) -> None:
        """ """

        self.wvec = wvec
        self.sEvec_c = sEvec_c
        self.beta = beta
        self.mu = mu 
        self.prefactor = 1.0/beta
        self.t = 1.0
        self.tp = 0.4
        return None

    
    def vz2_integrated(self) -> float:
        """velocity in z direction squared integrated"""
        return (2.0)


    def dfd_dw(self, omega: float) -> float:
        """derivative of fermi-dirac  with respect to omega"""
        
        beta: float = self.beta
        result: float
        if np.abs(beta*omega) > 50:
            result = 0.0
        else:
            result = -beta * np.exp(beta*omega)/(1.0 + np.exp(beta*omega))**2.0
        
        return result

    
    def calc_sigmadc(self) -> float:
        """ """
        sigma_dc: float  = 0.0
        Akw2 = periodize.periodize_Akw2
        y1 = periodize.Y1Limit
        y2 = periodize.Y2Limit

        integrand_w = np.zeros(self.wvec.shape)
        for (i, ww) in enumerate(self.wvec):
            integrand_w[i] = sI.dblquad(Akw2, -np.pi, np.pi, y1, y2, epsabs=1e-8,
                                        args=(self.t, self.tp, self.sEvec_c[i], ww, self.mu) )[0]
            integrand_w[i] *= self.dfd_dw(ww)
            
        sigma_dc = sI.simps(integrand_w, self.wvec)

        # Finalement, multiplication par toutes les constantes: mesures d'integration, etc.
        sigma_dc *= -1.0/(2.0*np.pi)**(3.0)*self.prefactor*self.vz2_integrated()

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

        sigma_dc = -1.0/(2.0*np.pi)*self.prefactor * self.vz2_integrated() * sI.simps(Aw2_vec * dfd_dw_vec, wwvec)

        return sigma_dc

if __name__ == "__main__":

    import os, json


    with open("statsparams.json") as fin:
        params = json.load(fin)
        mu = params["mu"][0]
        beta = params["beta"][0]
    
    for i in range(10):
        
        fname: str = "self_ctow" + str(i) + ".dat"
        if not os.path.isfile(fname):
            break

        (wvec, sEvec_c) = green.read_green_c(fname)
        SDC = SigmaDC(wvec, sEvec_c, beta=beta, mu=mu)
        SDC.calc_sigmadc()

