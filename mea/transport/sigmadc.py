import numpy as np  # type: ignore
from ..model import periodize  # type: ignore
import scipy.integrate as sI  # type: ignore
from ..tools import fmanip


class SigmaDC:

    def __init__(self, model, beta: float) -> None:
        """ """

        self.beta = beta
        self.prefactor = 2.0  # sum on spins,  weistrass-th, v_kz integrated
        self.cutoff = 30  # beta*omega
        self.model = model

        return None

    def dfd_dw(self, omega: float) -> float:
        """derivative of fermi-dirac  with respect to omega"""
        # implement fermi function as the exemple of the logistic function

        beta: float = self.beta
        result: float = -beta * \
            np.exp(beta * omega) / (1.0 + np.exp(beta * omega))**2.0
        return result

    def y1(self, x: float) -> float:
        return -np.pi

    def y2(self, x: float) -> float:
        return np.pi

    def calc_sigmadc(self) -> float:

        sigma_dc: float = 0.0
        Akw2 = self.model.periodize_Akw2
        Akw2_cum = self.model.periodize_Akw2_cum
        Akw2_trace = self.model.Akw2_trace

        integrand_w = np.zeros(self.model.z_vec.shape)
        integrand_w_cum = np.zeros(self.model.z_vec.shape)
        integrand_w_trace = np.zeros(self.model.z_vec.shape)
        for (i, ww) in enumerate(self.model.z_vec):

            if abs(self.beta * ww) > self.cutoff:
                integrand_w[i] = 0.0
                integrand_w_cum[i] = 0.0
            else:
                integrand_w[i] = 1.0 / (2.0 * np.pi)**(2.0) * sI.dblquad(Akw2, -np.pi, np.pi, self.y1, self.y2, epsabs=1e-8,
                                                                         args=(i,))[0]
                integrand_w_cum[i] = 1.0 / (2.0 * np.pi)**(2.0) * sI.dblquad(Akw2_cum, -np.pi, np.pi, self.y1, self.y2, epsabs=1e-8,
                                                                             args=(i,))[0]
                N_c = 4.0
                integrand_w_trace[i] = N_c / (2.0 * np.pi)**(2.0) * sI.dblquad(Akw2_trace, -np.pi / 2.0, np.pi / 2.0,
                                                                               lambda x: -np.pi / 2.0, lambda x: np.pi / 2.0, epsabs=1e-8,
                                                                               args=(i,))[0]
                dfd_dw = self.dfd_dw(ww)
                integrand_w[i] *= -dfd_dw
                integrand_w_cum[i] *= -dfd_dw
                integrand_w_trace[i] *= -dfd_dw

        sigma_dc = 1.0 / (2.0 * np.pi) * np.array([[sI.simps(
            integrand_w, self.model.z_vec), sI.simps(integrand_w_cum, self.model.z_vec), 0.5 * sI.simps(integrand_w_trace, self.model.z_vec)]])
        sigma_dc *= self.prefactor

        print("sigma_dc = ", sigma_dc)
        with open("sigmadc.dat", mode="a") as fout:
            for sigma in sigma_dc.flatten():
                fout.write(str(sigma) + " ")
            fout.write("\n")

        return sigma_dc

    def calc_sigmadc_test(self, Aw) -> float:
        """ calculate sigma_dc with a non-interacting A(w)given, independant of k.
            A(k,w) = A(w)(2pi)**2.0*delta(k)
            """

        sigma_dc: float = 0.0
        Aw2_vec = Aw[:, 1] * Aw[:, 1]
        wwvec = Aw[:, 0]

        dfd_dw_vec = np.zeros(wwvec.shape)
        for (i, ww) in enumerate(wwvec):
            dfd_dw_vec[i] = self.dfd_dw(ww)

        sigma_dc = -1.0 / (2.0 * np.pi) * self.prefactor * \
            sI.simps(Aw2_vec * dfd_dw_vec, wwvec)

        return sigma_dc

    def calc_sigmadc_interpolation(self)-> float:
        """preliminary and not tested. """

        sigma_dc: float = 0.0

        def Akw_0(kx, ky):
            im_gf = np.zeros(3)
            for jj in range(0, 3):
                im_gf[jj] = self.model.periodize_Akw(kx, ky, jj)
            return (np.polyfit(self.model.z_vec[0:3].imag, im_gf, 2)[-1])

        sigma_dc += 1.0 / (2.0 * np.pi)**(2.0) * sI.dblquad(Akw_0, -
                                                            np.pi, np.pi, self.y1, self.y2, epsabs=1e-8)[0]
        sigma_dc *= 1.0 / (2.0 * np.pi)
        sigma_dc *= self.prefactor

        print("sigma_dc = ", sigma_dc)

        return sigma_dc
