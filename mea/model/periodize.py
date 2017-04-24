from numba import jitclass, jit, double, complex128 # type: ignore
import numpy as np  # type: ignore
from scipy import linalg # type: ignore
from scipy.integrate import dblquad # type: ignore

class Model:
    """ """
    def __init__(self, t, tp, mu, w_vec, sEvec_c):
        """ """

        self.t = t ; self.tp = tp; self.mu = mu ;
        self.w_vec = w_vec ; self.sEvec_c = sEvec_c

#@jit
def t_value(kx, ky, t, tp): # This is t_ij(k_tilde)
    """ """

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


#@jit
def exp_k(kx, ky):

    expk = np.zeros(4, dtype=complex) # Here exp_k[i] is in fact e**(-j*dot(k, r_i)) where r_i is a site of the cluster
    expk[0] = 1.0
    expk[1] = np.exp(1.0j*kx)
    expk[2] = np.exp(1.0j*(kx+ky))
    expk[3] = np.exp(1.0j*ky)
    return expk


def build_gf_ktilde(kx, ky, t, tp, sE, w, mu):
    """ """
    gf_ktilde = np.zeros((4,4), dtype=complex)
    gf_ktilde = linalg.inv((w + mu) * np.eye(4) - t_value(kx, ky, t, tp) - sE)
    return gf_ktilde

def build_gf_ktilde_inverse(kx, ky, t, tp, sE, w, mu):
    """ """
    gf_ktilde_inverse = np.zeros((4,4), dtype=complex)
    gf_ktilde_inverse = (w + mu) * np.eye(4) - t_value(kx, ky, t, tp) - sE
    return gf_ktilde_inverse

#@jit
def periodize_Akw(kx, ky, t, tp, sE, w, mu): # periodize the imaginary part (Ak_w)
    """ """

    k = np.array([kx, ky])
    N_c = 4
    r_sites = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    sE = sE.copy()
    gf_ktilde = build_gf_ktilde(kx, ky, t, tp, sE, w, mu)
    expk = exp_k(kx, ky)
    gf_w_lattice = 1/N_c * np.dot(np.conjugate(expk), np.dot(gf_ktilde, expk))

    return (-2.0*gf_w_lattice.imag)

def periodize_Akw2(kx, ky, t, tp, sE, w, mu):
    return (periodize_Akw(kx, ky, t, tp, sE, w, mu)**2.0)


#@jit
def eps_0(kx, ky, t, tp):
    """the free dispersion relation for the given Model"""

    return -2.0*t*(np.cos(kx) + np.cos(ky)) - 2.0*tp*np.cos(kx + ky)


#@jit
def hopping_test(kx, ky, t, tp):
    """A different approach to calculate t_value (t(ktilde)) using the explicit fourier transform """

    k = np.array([kx, ky])
    N_c = 4
    r_sites = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    K_sites = np.array([[0.0, 0.0], [np.pi, 0.0], [np.pi, np.pi], [0.0, np.pi]])
    t_array = np.zeros((N_c, N_c), dtype=complex)

    for i in range(N_c):
        for j in range(N_c):
            for K in K_sites:
                t_array[i, j] += 1/N_c * np.exp(1.0j*np.dot(K + k, r_sites[i] - r_sites[j])) * eps_0(*(K + k), t, tp)

    return t_array


#@jit
def Y1Limit(x):
    return -np.pi


#@jit
def Y2Limit(x):
    return np.pi


def calc_dos(model, fout_name="dos.txt"):
    """ """
    sEvec_c = model.sEvec_c; w_vec = model.w_vec; t = model.t ; tp = model.tp; mu = model.mu
    len_sEvec_c = sEvec_c.shape[0]
    dos = np.zeros(len_sEvec_c)

    for n in range(len_sEvec_c):
        #print("IN LOOP of dos # ", n, " out of ", len_sEvec_c, "\n")
        dos[n] = (2.0*np.pi)**(-2.0)*dblquad(periodize_Akw, -np.pi, np.pi, Y1Limit, Y2Limit, args=(t, tp, sEvec_c[n], w_vec[n], mu))[0]
    dos_out = np.transpose([w_vec, dos])
    np.savetxt(fout_name, dos_out)
    return dos

#@jit
def Akw_trace(kx, ky, t, tp, sE, w, mu):
    """ """
    gf_ktilde = build_gf_ktilde(kx, ky, t, tp, sE, w , mu)
    Akw = -2.0*np.trace(gf_ktilde).imag
    return (Akw / 4.0)

def calc_dos_with_trace(model, fout_name="dos_trace.txt"):
    """ """
    sEvec_c = model.sEvec_c; w_vec = model.w_vec; t = model.t ; tp = model.tp; mu = model.mu
    len_sEvec_c = sEvec_c.shape[0]
    dos = np.zeros(len_sEvec_c)
    for n in range(len_sEvec_c):
        #print("IN LOOP of dos # ", n, " out of ", len_sEvec_c, "\n")
        dos[n] = (2.0*np.pi)**(-2.0)*dblquad(Akw_trace, -np.pi, np.pi, Y1Limit, Y2Limit, args=(t, tp, sEvec_c[n], w_vec[n], mu))[0]
    dos_out = np.transpose([w_vec, dos])
    np.savetxt(fout_name, dos_out)
    return dos


def periodize_Gkw_vec(model, kx, ky): # periodize the green function for all frequencies for one k
    """ """
    t = model.t; tp = model.tp; sEvec_c = model.sEvec_c; w_vec = model.w_vec; mu = model.mu
    k = np.array([kx, ky])
    N_c = 4
    r_sites = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    gf_ktilde_vec = sEvec_c.copy()
    for (i, sE) in enumerate(sEvec_c):
        gf_ktilde_vec[i]  = build_gf_ktilde(kx, ky, t, tp, sE, w_vec[i], mu)

    expk = exp_k(kx, ky)
    gf_lattice_vec = np.zeros(w_vec.shape[0], dtype=complex)

    for i in range(w_vec.shape[0]):
        gf_lattice_vec[i] = 1/N_c * np.dot(np.conjugate(expk), np.dot(gf_ktilde_vec[i], expk))

    return (gf_lattice_vec)


def build_sE_lattice_vec(model, kx, ky):
    """ """
    t = model.t; tp = model.tp; sEvec_c = model.sEvec_c; w_vec = model.w_vec; mu = model.mu

    gf_lattice_vec = periodize_Gkw_vec(model, kx, ky)
    sE_lattice_vec = np.copy(gf_lattice_vec)

    for (i, gf) in enumerate(gf_lattice_vec):
        sE_lattice_vec[i] = (w_vec[i] + mu) - eps_0(kx, ky, t, tp) - 1.0/gf

    return (sE_lattice_vec)


def zk_weight(model, kx, ky):
    """ """
    t = model.t; tp = model.tp; sEvec_c = model.sEvec_c; w_vec = model.w_vec; mu = model.mu
    sEvec_w = build_sE_lattice_vec(model, kx, ky)
    np.savetxt("sEw_Periodized.dat", np.transpose([w_vec, sEvec_w.real, sEvec_w.imag]))
    w0 = eps_0(kx, ky, t, tp) - mu
    indicesgreater = np.where(w_vec > w0)[0]
    indicesless = np.where(w_vec < w0)[0]
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
    zk1 = 1.0/(1.0 - derivative1)
    zk2 = 1.0/(1.0 - derivative2)
    return ((zk1, zk2))


def fermi_surface(model, w_value, fout="fermi_surface.dat"):
    """ """
    sEvec_c = model.sEvec_c; w_vec = model.w_vec; t = model.t ; tp = model.tp; mu = model.mu
    #find index of w_value
    w_idx = np.where(w_vec > w_value)[0][0]
    #print("w_idx = ", w_idx)
    kxarr = np.linspace(-np.pi, np.pi, 100)
    kyarr = kxarr.copy()
    Akw_vec = np.zeros((kxarr.shape[0]*kyarr.shape[0], 3))

    nb_w = 5
    ss = slice(w_idx - nb_w, w_idx + nb_w, 1)
    #print("ss = ", ss)
    #print("w_vec = ", w_vec)
    #print("w_vec.shape = ", w_vec.shape)
    w_range = w_vec[ss]
    #print("w_range = ", w_range)
    #print("w_range.shape = ", w_range.shape)
    sE_range = sEvec_c[ss]


    for(itt,(ww, sE)) in enumerate(zip(w_range, sE_range)):
        idx = 0
        for (i, kx) in enumerate(kxarr):
            for(j, ky) in enumerate(kyarr):
                Akw_vec[idx, 0] += kx
                Akw_vec[idx, 1] += ky
                Akw_vec[idx, 2] += periodize_Akw(kx, ky, t, tp, sE, ww, mu)
                idx+=1
    Akw_vec = 1.0/(w_range.shape[0])*Akw_vec.copy()
    np.savetxt(fout, Akw_vec)
