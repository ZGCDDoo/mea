from numba import jitclass, jit, double, complex128
import numpy as np
from scipy import linalg
from scipy.integrate import dblquad
from . import periodize



class Model:
    """ """
    def __init__(self, t, tp, mu, w_vec, sEvec_c):
        """ """

        self.t = t ; self.tp = tp; self.mu = mu ; 
        self.w_vec = w_vec ; self.sEvec_c = sEvec_c

#@jit
def t_value(kx, ky, t, tp): # This is t_ij(k_tilde)
    """ """
    
    t_value_up = np.zeros((4, 4), dtype=complex)
    ex = np.exp(-2.0j*kx) ; emx = np.conjugate(ex)
    ey = np.exp(-2.0j*ky) ; emy = np.conjugate(ey)
    tloc_up = np.array([[0.0, -t, -tp, -t],
                         [-t, 0.0, -t, 0.0],
                         [-tp, -t, 0.0, -t],
                         [-t, 0.0, -t, 0.0]])


    t_value_up += tloc_up

    t_value_up[0, 0] += 0.0;               t_value_up[0, 1] += -t*ex;               t_value_up[0, 2] += -tp*ex*ey;    t_value_up[0, 3] += -t*ey
    t_value_up[1, 0] += -t*emx;            t_value_up[1, 1] += 0.0;                 t_value_up[1, 2] += -t*ey;        t_value_up[1, 3] += -tp*(emx + ey)
    t_value_up[2, 0] += -tp*emx*emy;       t_value_up[2, 1] += -t*emy;              t_value_up[2, 2] += 0.0;          t_value_up[2, 3] += -t*emx
    t_value_up[3, 0] += -t*emy;            t_value_up[3, 1] += -tp*(ex + emy);      t_value_up[3, 2] += -t*ex;        t_value_up[3, 3] += 0.0

    t_value_down = -t_value_up.copy()
    zeros = np.zeros((4,4), dtype=complex)
    tmp1 = np.concatenate((t_value_up, zeros), axis=0)
    tmp2 = np.concatenate((zeros, t_value_down), axis=0)
    t_value = np.concatenate((tmp1, tmp2), axis=1)
    
    return (t_value)




def build_gf_ktilde(kx, ky, t, tp, sE, w, mu):
    """ """
    gf_ktilde = np.zeros((8,8), dtype=complex)
    gf_ktilde[0, 0] = gf_ktilde[1, 1] = gf_ktilde[2, 2] = gf_ktilde[3, 3] = (w + mu)
    gf_ktilde[4, 4] = gf_ktilde[5, 5] = gf_ktilde[6, 6] = gf_ktilde[7, 7] = -np.conjugate(w + mu)
    gf_ktilde -= t_value(kx, ky, t , tp)
    gf_ktilde -= sE
    gf_ktilde = linalg.inv(gf_ktilde.copy())

    return gf_ktilde       


#@jit
def Y1Limit(x):
    return -np.pi


#@jit
def Y2Limit(x):
    return np.pi        


#@jit
def Akw_trace(kx, ky, t, tp, sE, w, mu):
    """ """
    gf_ktilde = build_gf_ktilde(kx, ky, t, tp, sE, w , mu)
    Akw = -2.0*np.trace(gf_ktilde).imag
    return (Akw / 4.0)


def dos_with_trace(model, fout_name="dos_trace.txt"):
    """ """
    sEvec_c = model.sEvec_c; w_n = model.w_vec; t = model.t ; tp = model.tp; mu = model.mu
    len_sEvec_c = sEvec_c.shape[0]
    dos = np.zeros(len_sEvec_c)
    for n in range(len_sEvec_c):
        print("IN LOOP of dos # ", n, " out of ", len_sEvec_c, "\n")
        dos[n] = (2.0*np.pi)**(-2.0)*dblquad(Akw_trace, -np.pi, np.pi, Y1Limit, Y2Limit, args=(t, tp, sEvec_c[n], w_vec[n], mu))[0]
    dos_out = np.transpose([w_n, dos])
    np.savetxt(fout_name, dos_out)
    return dos



def periodize_nambu(kx, ky, t, tp, sE, w, mu): # Green periodization
    """ """
    nambu_ktilde = build_gf_ktilde(kx, ky, t, tp, sE, w, mu)
    ex = np.exp(1.0j*kx)
    ey = np.exp(1.0j*ky)
    v = np.array([1., ex, ex*ey, ey], dtype=complex)
    
    nambu_periodized = np.zeros((2, 2), dtype=complex)
        
    for  i in range(4):
        for j in range(4):
            nambu_periodized[0, 0] += np.conj(v[i])*nambu_ktilde[i, j]*v[j]
            nambu_periodized[0, 1] += np.conj(v[i])*nambu_ktilde[i, j + 4]*v[j]
            nambu_periodized[1, 0] += np.conj(v[i])*nambu_ktilde[i + 4, j]*v[j]
            nambu_periodized[1, 1] += np.conj(v[i])*nambu_ktilde[i + 4, j + 4]*v[j]
    
    
    nambu_periodized = 0.25*nambu_periodized

    return nambu_periodized

def stiffness(kx, ky, t, tp, sE, w, mu):
    """ """
    nambu_periodized = periodize_nambu(kx, ky, t, tp, sE, w, mu)
    coskx = np.cos(kx) 
    cosky = np.cos(ky)
    #tperp = -(coskx - cosky)*(coskx - cosky) # t_perp = -1.0
    #tperp_squared = tperp*tperp
    tperp_squared = 2.0
    N_c = 4.0
    return (-1.0 * np.real(-4.0*tperp_squared*nambu_periodized[0, 1]*nambu_periodized[1, 0]))



def matsubara_surface(model, fout="matsubara_surface.dat"):
    """Plot the norm of the gorkov function as a function of kx, ky."""
    sEvec_c = model.sEvec_c; w_vec = model.w_vec; t = model.t ; tp = model.tp; mu = model.mu
    kxarr = np.linspace(-np.pi, np.pi, 100)
    kyarr = kxarr.copy()
    Akw_vec = np.zeros((kxarr.shape[0]*kyarr.shape[0], 3))

    idx = 0; sE = sEvec_c[0] ; ww = w_vec[0]
    for (i, kx) in enumerate(kxarr):
        for(j, ky) in enumerate(kyarr):
            Akw_vec[idx, 0] += kx
            Akw_vec[idx, 1] += ky
            Akw_vec[idx, 2] += np.absolute(periodize_nambu(kx, ky, t, tp, sE, ww, mu)[0, 1])
            idx+=1

    np.savetxt(fout, Akw_vec)


