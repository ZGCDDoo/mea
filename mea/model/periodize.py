import numpy as np  # type: ignore
from scipy import linalg # type: ignore
from scipy.integrate import dblquad # type: ignore
from typing import Tuple



class Model:
    """ """
    def __init__(self, t: float, tp: float, mu: float, z_vec, sEvec_c) -> None:
        """ """

        assert z_vec.shape[0] == sEvec_c.shape[0]
        self.t = t ; self.tp = tp; self.mu = mu 
        self.z_vec = z_vec ; self.sEvec_c = sEvec_c


    def t_value(self, kx: float, ky: float) : # This is t_ij(k_tilde)
        """ """
        t = self.t ; tp = self.tp
        t_val = np.zeros((4, 4), dtype=complex)
        ex = np.exp(-2.0j*kx) ; emx = np.conjugate(ex)
        ey = np.exp(-2.0j*ky) ; emy = np.conjugate(ey)
        tloc = np.array([[0.0, -t, -tp, -t],
                            [-t, 0.0, -t, 0.0],
                            [-tp, -t, 0.0, -t],
                            [-t, 0.0, -t, 0.0]])


        t_val += tloc

        t_val[0, 0] += 0.0;               t_val[0, 1] += -t*ex;               t_val[0, 2] += -tp*ex*ey;    t_val[0, 3] += -t*ey
        t_val[1, 0] += -t*emx;            t_val[1, 1] += 0.0;                 t_val[1, 2] += -t*ey;        t_val[1, 3] += -tp*(emx + ey)
        t_val[2, 0] += -tp*emx*emy;       t_val[2, 1] += -t*emy;              t_val[2, 2] += 0.0;          t_val[2, 3] += -t*emx
        t_val[3, 0] += -t*emy;            t_val[3, 1] += -tp*(ex + emy);      t_val[3, 2] += -t*ex;        t_val[3, 3] += 0.0

        return (t_val)



    def exp_k(self, kx: float, ky: float):

        expk = np.zeros(4, dtype=complex) # Here exp_k[i] is in fact e**(-j*dot(k, r_i)) where r_i is a site of the cluster
        expk[0] = 1.0
        expk[1] = np.exp(1.0j*kx)
        expk[2] = np.exp(1.0j*(kx+ky))
        expk[3] = np.exp(1.0j*ky)
        return expk


    
    def eps_0(self, kx: float, ky: float) -> float:
        """the free dispersion relation for the given Model"""
        t: float = self.t ; tp: float = self.tp
        return -2.0*t*(np.cos(kx) + np.cos(ky)) - 2.0*tp*np.cos(kx + ky)


    
    def hopping_test(self, kx: float, ky: float):
        """A different approach to calculate t_value (t(ktilde)) using the explicit fourier transform """

        k = np.array([kx, ky])
        N_c: int = 4
        r_sites = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        K_sites = np.array([[0.0, 0.0], [np.pi, 0.0], [np.pi, np.pi], [0.0, np.pi]])
        t_array = np.zeros((N_c, N_c), dtype=complex)

        for i in range(N_c):
            for j in range(N_c):
                for K in K_sites:
                    t_array[i, j] += 1/N_c * np.exp(1.0j*np.dot(K + k, r_sites[i] - r_sites[j])) * self.eps_0(*(K + k))

        return t_array



    def ky1Limit(self, x: float) -> float:
        return -np.pi



    def ky2Limit(self, x: float) -> float:
        return np.pi


    def build_gf_ktilde(self, kx: float, ky: float, ii: int):
        """ """
        w = self.z_vec[ii] ; sE = self.sEvec_c[ii] ; mu: float = self.mu
        gf_ktilde = np.zeros((4,4), dtype=complex)
        gf_ktilde = linalg.inv((w + mu) * np.eye(4) - self.t_value(kx, ky) - sE)
        return gf_ktilde


    def build_gf_ktilde_inverse(self, kx: float, ky: float, ii: int):
        """ """
        w = self.z_vec[ii] ; sE = self.sEvec_c[ii] ; mu: float = self.mu
        gf_ktilde_inverse = np.zeros((4,4), dtype=complex)
        gf_ktilde_inverse = (w + mu) * np.eye(4) - self.t_value(kx, ky) - sE
        return gf_ktilde_inverse


    def periodize_Akw(self, kx: float, ky: float, ii: int) -> float: # periodize the imaginary part (Ak_w)
        """ """
        N_c: int = 4
        gf_ktilde = self.build_gf_ktilde(kx, ky, ii)
        expk = self.exp_k(kx, ky)
        gf_w_lattice = 1/N_c * np.dot(np.conjugate(expk), np.dot(gf_ktilde, expk))

        return (-2.0*gf_w_lattice.imag)


    def periodize_Akw2(self, kx: float, ky: float, ii:int) -> float:
        return (self.periodize_Akw(kx, ky, ii)**2.0)


    def calc_dos(self, fout_name="dos.txt"):
        """ """
        sEvec_c = self.sEvec_c; z_vec = self.z_vec; t = self.t ; tp = self.tp; mu = self.mu
        len_sEvec_c = sEvec_c.shape[0]
        dos = np.zeros(len_sEvec_c)

        ii: int
        for ii in range(len_sEvec_c):
            #print("IN LOOP of dos # ", n, " out of ", len_sEvec_c, "\n")
            dos[ii] = (2.0*np.pi)**(-2.0)*dblquad(self.periodize_Akw, -np.pi, np.pi, self.ky1Limit, self.ky2Limit, args=(ii,))[0]
        dos_out = np.transpose([z_vec, dos])
        np.savetxt(fout_name, dos_out)
        return dos


    def Akw_trace(self, kx: float, ky: float, ii: int) -> float:
        """ """
        gf_ktilde = self.build_gf_ktilde(kx, ky, ii)
        Akw = -2.0*np.trace(gf_ktilde).imag
        return (Akw / 4.0)


    def calc_dos_with_trace(self, fout_name="dos_trace.txt"):
        """ """
        sEvec_c = self.sEvec_c; z_vec = self.z_vec; t = self.t ; tp = self.tp; mu = self.mu
        len_sEvec_c = sEvec_c.shape[0]
        dos = np.zeros(len_sEvec_c)
        for ii in range(len_sEvec_c):
            #print("IN LOOP of dos # ", n, " out of ", len_sEvec_c, "\n")
            dos[ii] = (2.0*np.pi)**(-2.0)*dblquad(self.Akw_trace, -np.pi, np.pi, self.ky1Limit, self.ky2Limit, args=(ii,))[0]
        dos_out = np.transpose([z_vec, dos])
        np.savetxt(fout_name, dos_out)
        return dos


    def periodize_Gkz_vec(self, kx: float, ky: float): # periodize the green function for all frequencies for one k
        """ """
        t = self.t; tp = self.tp; sEvec_c = self.sEvec_c; z_vec = self.z_vec; mu = self.mu
        k = np.array([kx, ky])
        N_c = 4
        gf_ktilde_vec = sEvec_c.copy()
        for (ii, sE) in enumerate(sEvec_c):
            gf_ktilde_vec[ii]  = self.build_gf_ktilde(kx, ky, ii)

        expk = self.exp_k(kx, ky)
        gf_lattice_vec = np.zeros(z_vec.shape[0], dtype=complex)

        for i in range(z_vec.shape[0]):
            gf_lattice_vec[i] = 1/N_c * np.dot(np.conjugate(expk), np.dot(gf_ktilde_vec[i], expk))

        return (gf_lattice_vec)


    def build_sE_lattice_vec(self, kx: float, ky: float):
        """ """
        t = self.t; tp = self.tp; sEvec_c = self.sEvec_c; z_vec = self.z_vec; mu = self.mu

        gf_lattice_vec = self.periodize_Gkz_vec(kx, ky)
        sE_lattice_vec = np.copy(gf_lattice_vec)

        for (i, gf) in enumerate(gf_lattice_vec):
            sE_lattice_vec[i] = (z_vec[i] + mu) - self.eps_0(kx, ky) - 1.0/gf

        return (sE_lattice_vec)


    def zk_weight(self, kx: float, ky: float) -> Tuple[float, float]:
        """ """
        t = self.t; tp = self.tp; sEvec_c = self.sEvec_c; z_vec = self.z_vec; mu = self.mu
        sEvec_w = self.build_sE_lattice_vec(kx, ky)
        np.savetxt("sEw_Periodized.dat", np.transpose([z_vec, sEvec_w.real, sEvec_w.imag]))
        w0: float = self.eps_0(kx, ky) - mu
        indicesgreater = np.where(z_vec > w0)[0]
        indicesless = np.where(z_vec < w0)[0]
        grid1 = np.array([indicesless[-1], indicesgreater[0], indicesgreater[1]])
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
        derivative1 = 2.0*w0*coefs1[0] + coefs1[1]
        derivative2 = 2.0*w0*coefs2[0] + coefs2[1]
        zk1: float = 1.0/(1.0 - derivative1)
        zk2: float = 1.0/(1.0 - derivative2)
        return ((zk1, zk2))


    def fermi_surface(self, w_value, fout="fermi_surface.dat"):
        """ """
        sEvec_c = self.sEvec_c; z_vec = self.z_vec; t = self.t ; tp = self.tp; mu = self.mu
        #find index of w_value
        w_idx = np.where(z_vec > w_value)[0][0]
        #print("w_idx = ", w_idx)
        kxarr = np.linspace(-np.pi, np.pi, 100)
        kyarr = kxarr.copy()
        Akz_vec = np.zeros((kxarr.shape[0]*kyarr.shape[0], 3))

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
                    Akz_vec[idx, 2] += self.periodize_Akw(kx, ky, t, tp, sE, ww, mu)
                    idx+=1
        Akz_vec = 1.0/(w_range.shape[0])*Akz_vec.copy()
        np.savetxt(fout, Akz_vec)
