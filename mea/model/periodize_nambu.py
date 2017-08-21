import numpy as np
from scipy import linalg
from scipy.integrate import dblquad



class ModelNambu: 
    """ """
    def __init__(self, t: float, tp: float, mu: float, z_vec, sEvec_c) -> None:
        """ """

        self.t = t ; self.tp = tp; self.mu = mu ; 
        self.z_vec = z_vec ; self.sEvec_c = sEvec_c
        self.cumulants = self.build_cumulants()
        return None


    def t_value(self, kx: float, ky: float): # This is t_ij(k_tilde)
        """ """
        (t, tp) = (self.t, self.tp)
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




    def build_gf_ktilde(self, kx: float, ky: float, ii: int):
        """ """
        gf_ktilde = np.zeros((8,8), dtype=complex)
        (zz, mu, sE) = (self.z_vec[ii], self.mu, self.sEvec_c[ii])
        gf_ktilde[0, 0] = gf_ktilde[1, 1] = gf_ktilde[2, 2] = gf_ktilde[3, 3] = (zz + mu)
        gf_ktilde[4, 4] = gf_ktilde[5, 5] = gf_ktilde[6, 6] = gf_ktilde[7, 7] = -np.conjugate(-np.conjugate(zz) + mu)
        gf_ktilde -= self.t_value(kx, ky)
        gf_ktilde -= sE
        gf_ktilde = linalg.inv(gf_ktilde.copy())

        return gf_ktilde       


    def build_cumulants(self):
        """ """

        cumulants = np.zeros(self.sEvec_c.shape, dtype=complex)
        for ii in range(cumulants.shape[0]):
            tmp = np.zeros((8, 8), dtype=complex)
            (zz, mu, sE) = (self.z_vec[ii], self.mu, self.sEvec_c[ii])
            tmp[0, 0] = tmp[1, 1] = tmp[2, 2] = tmp[3, 3] = (zz + mu)
            tmp[4, 4] = tmp[5, 5] = tmp[6, 6] = tmp[7, 7] = -np.conjugate(-np.conjugate(zz) + mu)
            tmp -= sE
            tmp = linalg.inv(tmp.copy())
            cumulants[ii] = tmp.copy()
        
        return cumulants    

    
    def Y1Limit(self, x: float) -> float:
        return -np.pi


    
    def Y2Limit(self, x: float) -> float:
        return np.pi        


    
    def Akw_trace(self, kx: float, ky: float, ii: int):
        """ """
        gf_ktilde = self.build_gf_ktilde(kx, ky, ii)
        Akw = -2.0*np.trace(gf_ktilde).imag
        return (Akw / 4.0)


    def dos_with_trace(self, fout_name="dos_trace_nambu.dat"):
        """ """
        len_sEvec_c: int = self.sEvec_c.shape[0]
        dos = np.zeros(len_sEvec_c)
        for n in range(len_sEvec_c):
            print("IN LOOP of dos # ", n, " out of ", len_sEvec_c, "\n")
            dos[n] = (2.0*np.pi)**(-2.0)*dblquad(self.Akw_trace, -np.pi, np.pi, self.Y1Limit, self.Y2Limit, args=(n,))[0]
        dos_out = np.transpose([self.z_vec, dos])
        np.savetxt(fout_name, dos_out)
        return dos


    def periodize(self, kx: float, ky: float, arg):
        """ """
        
        ex = np.exp(1.0j*kx)
        ey = np.exp(1.0j*ky)
        v = np.array([1., ex, ex*ey, ey], dtype=complex)
        nambu_periodized = np.zeros((2, 2), dtype=complex)

        for  i in range(4):
            for j in range(4):
                nambu_periodized[0, 0] += np.conj(v[i])*arg[i, j]*v[j]
                nambu_periodized[0, 1] += np.conj(v[i])*arg[i, j + 4]*v[j]
                nambu_periodized[1, 0] += np.conj(v[i])*arg[i + 4, j]*v[j]
                nambu_periodized[1, 1] += np.conj(v[i])*arg[i + 4, j + 4]*v[j]
        
        return (0.25*nambu_periodized)


    def periodize_nambu(self, kx: float, ky: float, ii: int): # Green periodization
        """ """
        nambu_ktilde = self.build_gf_ktilde(kx, ky, ii)

        return self.periodize(kx, ky, nambu_ktilde) 


    def stiffness(self, kx: float, ky: float, ii: int) -> float:
        """ """
        nambu_periodized = self.periodize(kx, ky, self.build_gf_ktilde(kx, ky, ii)) #self.periodize_nambu(kx, ky, ii)
        #coskx: float = np.cos(kx) 
        #cosky: float = np.cos(ky)
        #tperp = -(coskx - cosky)*(coskx - cosky) # t_perp = -1.0
        #tperp_squared = tperp*tperp
        #N_c = 4.0
        tperp_squared = 2.0
        return (-1.0 * np.real(-4.0*tperp_squared*nambu_periodized[0, 1]*nambu_periodized[1, 0]))


    def periodize_cumulant(self, kx: float, ky: float, ii: int): # cumulant periodization
        """ """
        tmp = linalg.inv(self.periodize(kx, ky, self.cumulants[ii]))
        eps = -2.0*self.t*(np.cos(kx) + np.cos(ky)) - 2.0*self.tp*np.cos(kx + ky)
        tmp[0, 0] -= eps; tmp[1, 1] += eps
        return linalg.inv(tmp)


    def stiffness_cum(self, kx: float, ky: float, ii: int) -> float:
        """ """
        
        nambu_periodized = self.periodize_cumulant(kx, ky, ii) #linalg.inv(tmp.copy())
        #coskx: float = np.cos(kx) 
        #cosky: float = np.cos(ky)
        #tperp = -(coskx - cosky)*(coskx - cosky) # t_perp = -1.0
        #tperp_squared = tperp*tperp
        #N_c = 4.0
        tperp_squared = 2.0
        return (-1.0 * np.real(-4.0*tperp_squared*nambu_periodized[0, 1]*nambu_periodized[1, 0]))    

    def stiffness_trace(self, kx: float, ky: float, ii: int) -> float:
        """4/N_c Trace(F F^Dag) """
        gf_ktilde = self.build_gf_ktilde(kx, ky, ii)
        trace = np.trace(np.dot(gf_ktilde[:4:, 4::], gf_ktilde[4::, :4:]))    
        tperp_squared = 2.0
        return (tperp_squared*np.real(trace))
    
    def matsubara_surface(self, fout: str ="matsubara_surface.dat"):
        """Plot the norm of the gorkov function as a function of kx, ky."""
        
        kxarr = np.linspace(-np.pi, np.pi, 100)
        kyarr = kxarr.copy()
        Akz_vec = np.zeros((kxarr.shape[0]*kyarr.shape[0], 3))

        idx: int = 0
        for (i, kx) in enumerate(kxarr):
            for(j, ky) in enumerate(kyarr):
                Akz_vec[idx, 0] += kx
                Akz_vec[idx, 1] += ky
                Akz_vec[idx, 2] += np.absolute(self.periodize_nambu(kx, ky, idx)[0, 1])
                idx+=1

        np.savetxt(fout, Akz_vec)


