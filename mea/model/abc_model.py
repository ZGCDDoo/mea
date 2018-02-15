import numpy as np  # type: ignore
from scipy import linalg  # type: ignore
from scipy.integrate import dblquad  # type: ignore
from typing import Tuple
import abc


class ABCModel(abc.ABC):
    """ """

    def __init__(self, t: float, tp: float, tpp: float,  mu: float, z_vec, sEvec_c) -> None:
        """ """

        assert z_vec.shape[0] == sEvec_c.shape[0]
        self.t = t
        self.tp = tp
        self.tpp = tpp
        self.mu = mu
        self.z_vec = z_vec
        self.sEvec_c = sEvec_c
        self.cumulants = self.build_cumulants()

        self._Nc = self.sEvec_c.shape[1]

        return None

    def build_cumulants(self):
        """ """
        cumulants = np.zeros(self.sEvec_c.shape, dtype=complex)
        for ii in range(cumulants.shape[0]):
            (zz, mu, sE) = (self.z_vec[ii], self.mu, self.sEvec_c[ii])
            tmp = np.eye(sE.shape[0], dtype=complex)
            tmp *= (zz + mu)
            tmp -= sE
            tmp = linalg.inv(tmp.copy())
            cumulants[ii] = tmp.copy()

        return cumulants

    @abc.abstractmethod
    def t_value(self, kx: float, ky: float):  # This is t_ij(k_tilde)
        """ """
        return None

    @abc.abstractmethod
    def eps_0(self, kx: float, ky: float) -> float:
        """the free dispersion relation for the given Model"""
        return None

    @abc.abstractmethod
    def hopping_test(self, kx: float, ky: float):
        """A different approach to calculate t_value (t(ktilde)) using the explicit fourier transform """
        return None

    def exp_k(self, kx: float, ky: float):

        # Here exp_k[i] is in fact e**(-j*dot(k, r_i)) where r_i is a site of the cluster
        expk = np.zeros(4, dtype=complex)
        expk[0] = 1.0
        expk[1] = np.exp(1.0j * kx)
        expk[2] = np.exp(1.0j * (kx + ky))
        expk[3] = np.exp(1.0j * ky)
        return expk

    def ky1Limit(self, x: float) -> float:
        return -np.pi

    def ky2Limit(self, x: float) -> float:
        return np.pi

    def build_gf_ktilde(self, kx: float, ky: float, ii: int):
        """ """
        w = self.z_vec[ii]
        sE = self.sEvec_c[ii]
        mu: float = self.mu
        gf_ktilde = np.zeros((4, 4), dtype=complex)
        gf_ktilde = linalg.inv((w + mu) * np.eye(4) -
                               self.t_value(kx, ky) - sE)
        return gf_ktilde

    def build_gf_ktilde_inverse(self, kx: float, ky: float, ii: int):
        """ """
        w = self.z_vec[ii]
        sE = self.sEvec_c[ii]
        mu: float = self.mu
        gf_ktilde_inverse = np.zeros((4, 4), dtype=complex)
        gf_ktilde_inverse = (w + mu) * np.eye(4) - self.t_value(kx, ky) - sE
        return gf_ktilde_inverse

    # periodize the imaginary part (Ak_w)
    def periodize_Akw(self, kx: float, ky: float, ii: int) -> float:
        """ """
        gf_ktilde = self.build_gf_ktilde(kx, ky, ii)
        gf_w_lattice = self.periodize(kx, ky, gf_ktilde)
        return (-2.0 * gf_w_lattice.imag)

    def periodize(self, kx: float, ky: float, arg) -> float:
        """ """
        N_c: int = self._Nc
        expk = self.exp_k(kx, ky)
        return(1 / N_c * np.dot(np.conjugate(expk), np.dot(arg, expk)))

    def periodize_Akw2(self, kx: float, ky: float, ii: int) -> float:
        return (self.periodize_Akw(kx, ky, ii)**2.0)

    def periodize_Akw_cum(self, kx: float, ky: float, ii: int) -> float:
        tmp = 1.0 / self.periodize(kx, ky, self.cumulants[ii])
        tmp -= self.eps_0(kx, ky)
        return ((-2.0 / tmp).imag)

    def periodize_Akw2_cum(self, kx: float, ky: float, ii: int) -> float:
        return (self.periodize_Akw_cum(kx, ky, ii)**2.0)

    def calc_dos(self, fout_name="dos.txt", fct="periodize_Akw"):
        """fct can be periodize_Akw or periodize_Akw_cum"""

        sEvec_c = self.sEvec_c
        z_vec = self.z_vec
        len_sEvec_c = sEvec_c.shape[0]
        dos = np.zeros(len_sEvec_c)

        ii: int
        for ii in range(len_sEvec_c):
            # print("IN LOOP of dos # ", n, " out of ", len_sEvec_c, "\n")
            dos[ii] = (2.0 * np.pi)**(-2.0) * dblquad(getattr(self, fct), -
                                                      np.pi, np.pi, self.ky1Limit, self.ky2Limit, args=(ii,))[0]
        dos_out = np.transpose([z_vec, dos])
        np.savetxt(fout_name, dos_out)
        return dos

    def Akw_trace(self, kx: float, ky: float, ii: int) -> float:
        """ """
        gf_ktilde = self.build_gf_ktilde(kx, ky, ii)
        Akw = -2.0 * np.trace(gf_ktilde).imag
        return (Akw / 4.0)

    def Akw2_trace(self, kx: float, ky: float, ii: int) -> float:
        """ """
        return (self.Akw_trace(kx, ky, ii)**2.0)

    def calc_dos_with_trace(self, fout_name="dos_trace.txt"):
        """ """
        sEvec_c = self.sEvec_c
        z_vec = self.z_vec
        len_sEvec_c = sEvec_c.shape[0]
        dos = np.zeros(len_sEvec_c)
        for ii in range(len_sEvec_c):
            # print("IN LOOP of dos # ", n, " out of ", len_sEvec_c, "\n")
            dos[ii] = (2.0 * np.pi)**(-2.0) * dblquad(self.Akw_trace, -
                                                      np.pi, np.pi, self.ky1Limit, self.ky2Limit, args=(ii,))[0]
        dos_out = np.transpose([z_vec, dos])
        np.savetxt(fout_name, dos_out)
        return dos

    # periodize the green function for all frequencies for one k
    def periodize_Gkz_vec(self, kx: float, ky: float):
        """ """
        sEvec_c = self.sEvec_c
        z_vec = self.z_vec
        mu = self.mu
        k = np.array([kx, ky])
        N_c = self._Nc

        gf_ktilde_vec = sEvec_c.copy()
        for (ii, sE) in enumerate(sEvec_c):
            gf_ktilde_vec[ii] = self.build_gf_ktilde(kx, ky, ii)

        expk = self.exp_k(kx, ky)
        gf_lattice_vec = np.zeros(z_vec.shape[0], dtype=complex)

        for i in range(z_vec.shape[0]):
            gf_lattice_vec[i] = 1 / N_c * \
                np.dot(np.conjugate(expk), np.dot(gf_ktilde_vec[i], expk))

        return (gf_lattice_vec)

    def build_sE_lattice_vec(self, kx: float, ky: float):
        """ """
        sEvec_c = self.sEvec_c
        z_vec = self.z_vec
        mu = self.mu

        gf_lattice_vec = self.periodize_Gkz_vec(kx, ky)
        sE_lattice_vec = np.copy(gf_lattice_vec)

        for (i, gf) in enumerate(gf_lattice_vec):
            sE_lattice_vec[i] = (z_vec[i] + mu) - self.eps_0(kx, ky) - 1.0 / gf

        return (sE_lattice_vec)

    def zk_weight(self, kx: float, ky: float) -> Tuple[float, float]:
        """ """
        sEvec_c = self.sEvec_c
        z_vec = self.z_vec
        mu = self.mu
        sEvec_w = self.build_sE_lattice_vec(kx, ky)
        np.savetxt("sEw_Periodized.dat", np.transpose(
            [z_vec, sEvec_w.real, sEvec_w.imag]))
        w0: float = 0.0  # self.eps_0(kx, ky) - mu
        indicesgreater = np.where(z_vec > w0)[0]
        indicesless = np.where(z_vec < w0)[0]
        grid1 = np.array(
            [indicesless[-1], indicesgreater[0], indicesgreater[1]])
        grid2 = np.array([indicesless[-2], indicesless[-1], indicesgreater[0]])
        fct1 = sEvec_w[grid1].real.copy()
        fct2 = sEvec_w[grid2].real.copy()
        coefs1 = np.polyfit(grid1, fct1, 2)
        coefs2 = np.polyfit(grid2, fct2, 2)
        #print("grid1 = ", grid1)
        #print("grid2 = ", grid2)
        #print("coefs1 = ", coefs1)
        #print("coefs2 = ", coefs2)
        #print("w0 = ", w0)
        derivative1 = 2.0 * w0 * coefs1[0] + coefs1[1]
        derivative2 = 2.0 * w0 * coefs2[0] + coefs2[1]
        zk1: float = 1.0 / (1.0 - derivative1)
        zk2: float = 1.0 / (1.0 - derivative2)
        zk_naif = 1.0 / (1.0 - (fct1[-1] - fct1[0]) / (grid1[-1] - grid1[0]))
        return ((zk_naif, zk1, zk2))

    def fermi_surface(self, w_value, fout="fermi_surface.dat"):
        """ """
        sEvec_c = self.sEvec_c
        z_vec = self.z_vec
        mu = self.mu
        # find index of w_value
        w_idx = np.where(z_vec > w_value)[0][0]
        #print("w_idx = ", w_idx)
        kxarr = np.linspace(-np.pi, np.pi, 100)
        kyarr = kxarr.copy()
        Akz_vec = np.zeros((kxarr.shape[0] * kyarr.shape[0], 3))

        nb_w = 5
        ss = slice(w_idx - nb_w, w_idx + nb_w, 1)
        #print("ss = ", ss)
        #print("z_vec = ", z_vec)
        #print("z_vec.shape = ", z_vec.shape)
        w_range = z_vec[ss]
        #print("w_range = ", w_range)
        #print("w_range.shape = ", w_range.shape)
        sE_range = sEvec_c[ss]

        for(itt, (ww, sE)) in enumerate(zip(w_range, sE_range)):
            idx = 0
            for (i, kx) in enumerate(kxarr):
                for(j, ky) in enumerate(kyarr):
                    Akz_vec[idx, 0] += kx
                    Akz_vec[idx, 1] += ky
                    Akz_vec[idx,
                            2] += self.periodize_Akw(kx, ky, t, tp, sE, ww, mu)
                    idx += 1
        Akz_vec = 1.0 / (w_range.shape[0]) * Akz_vec.copy()
        np.savetxt(fout, Akz_vec)
