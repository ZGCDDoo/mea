import numpy as np
from scipy import linalg
from scipy.integrate import dblquad


#For Olivier and qcm


#Copyright Charles-David Hebert
#MIT Licencse, use it as you see fit, but please give retributions to the author.


class ModelNambu: 
    """ """
    def __init__(self, t: float, tp: float, tpp:float, mu: float, z_vec, sEvec_c) -> None:
        """ """

        self.t = t ; self.tp = tp; self.tpp = tpp; self.mu = mu ; 
        self.z_vec = z_vec ; self.sEvec_c = sEvec_c
        self.cumulants = self.build_cumulants()
        return None



    def t_value(self, kx: float, ky: float) : # This is t_ij(k_tilde)
        """this t_value is only good if tpp = 0.0"""
        t = self.t ; tp = self.tp; tpp = self.tpp
        t_val_up = np.zeros((4, 4), dtype=complex)
        ex = np.exp(-2.0j*kx) ; emx = np.conjugate(ex)
        ey = np.exp(-2.0j*ky) ; emy = np.conjugate(ey)
        tloc = np.array([[0.0, -t, -t, -tp],
                         [-t, 0.0, 0.0, -t],
                         [-t, 0.0, 0.0, -t],
                         [-tp, -t, -t, 0.0]])


        t_val_up += tloc

        t_val_up[0, 0] += -tpp*(ex+emx+ey+emy);      t_val_up[0, 1] += -t*ex;                   t_val_up[0, 3] += -tp*(ex + ey + ex*ey);    t_val_up[0, 2] += -t*ey
        t_val_up[1, 0] += -t*emx;                    t_val_up[1, 1] += -tpp*(ex+emx+ey+emy);    t_val_up[1, 3] += -t*ey;                    t_val_up[1, 2] += -tp*(emx + ey + emx*ey)
        t_val_up[3, 0] += -tp*(emx + emy + emx*emy); t_val_up[3, 1] += -t*emy;                  t_val_up[3, 3] += -tpp*(ex+emx+ey+emy);     t_val_up[3, 2] += -t*emx
        t_val_up[2, 0] += -t*emy;                    t_val_up[2, 1] += -tp*(ex + emy + ex*emy); t_val_up[2, 3] += -t*ex;                    t_val_up[2, 2] += -tpp*(ex+emx+ey+emy)


        t_val_down = -t_val_up.copy()
        zeros = np.zeros((4,4), dtype=complex)
        tmp1 = np.concatenate((t_val_up, zeros), axis=0)
        tmp2 = np.concatenate((zeros, t_val_down), axis=0)
        t_val = np.concatenate((tmp1, tmp2), axis=1)

        return (t_val)





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



    def periodize(self, kx: float, ky: float, arg):
        """ """
        
        ex = np.exp(1.0j*kx)
        ey = np.exp(1.0j*ky)
        exQ = np.exp(1.0j*(kx+np.pi))
        eyQ = np.exp(1.0j*(ky+np.pi))
        vk = np.array([1.0, ex, ey, ex*ey], dtype=complex)
        vkQ = np.array([1.0, exQ, eyQ, exQ*eyQ], dtype=complex)
        nambu_periodized = np.zeros((4, 4), dtype=complex)

        gup = arg[:4:, :4:]
        gdown = arg[4::, 4::]
        ff = arg[:4:, 4::]
        ffdag = arg[4::, :4:]

        llgreen = [gup, ff, gdown, ffdag]

        llperiodized = [None]*4
        for ii in range(4):
            llperiodized[ii] =   np.array([
                                    [np.dot(np.conjugate(vk), np.dot(llgreen[ii], vk)), np.dot(np.conjugate(vk), np.dot(llgreen[ii], vkQ))],
                                    [np.dot(np.conjugate(vkQ), np.dot(llgreen[ii], vk)), np.dot(np.conjugate(vkQ), np.dot(llgreen[ii], vkQ))]
                                        ], 
                                    dtype=complex)
                                

        nambu_periodized[:2:, :2:] = llperiodized[0]
        nambu_periodized[:2:, 2::] = llperiodized[1]
        nambu_periodized[2::, 2::] = llperiodized[2]
        nambu_periodized[2::, :2:] = llperiodized[3]

        return (0.25*nambu_periodized)


    def periodize_orbitale(self, kx: float, ky: float, arg):
        
        r1 = np.array([0.0, 0.0])
        r2 = np.array([0.0, 1.0])
        ex = np.exp(1.0j*kx)
        vk = np.array([1.0, ex], dtype=complex)
        nambu_periodized = np.zeros((4, 4), dtype=complex)

        gup = arg[:4:, :4:]
        gdown = arg[4::, 4::]
        ff = arg[:4:, 4::]
        ffdag = arg[4::, :4:]

        llgreen = [gup, ff, gdown, ffdag]

        llperiodized = [None]*4
        for ii in range(4):

            block00 = np.array([
                            [llgreen[ii][0, 0], llgreen[ii][0, 3]],
                            [llgreen[ii][3, 0], llgreen[ii][3, 3]]
                            ])

            gAB00 = np.dot(np.conjugate(vk), np.dot(block00,vk))


            block01 = np.array([
                            [llgreen[ii][0, 1], llgreen[ii][0, 2]],
                            [llgreen[ii][3, 1], llgreen[ii][3, 2]]
                            ])

            gAB01 = np.dot(np.conjugate(vk), np.dot(block01,vk))


            block10 = np.array([
                            [llgreen[ii][1, 0], llgreen[ii][2, 0]],
                            [llgreen[ii][1, 3], llgreen[ii][2, 3]]
                            ])

            gAB10 = np.dot(np.conjugate(vk), np.dot(block10,vk))


            block11 = np.array([
                            [llgreen[ii][1, 1], llgreen[ii][1, 2]],
                            [llgreen[ii][2, 1], llgreen[ii][2, 2]]
                            ])

            gAB11 = np.dot(np.conjugate(vk), np.dot(block11,vk))

                                

            llperiodized[ii] = np.array([[gAB00, gAB01], [gAB10, gAB11]] ,dtype=complex)

        nambu_periodized[:2:, :2:] = llperiodized[0]
        nambu_periodized[:2:, 2::] = llperiodized[1]
        nambu_periodized[2::, 2::] = llperiodized[2]
        nambu_periodized[2::, :2:] = llperiodized[3]

        Nc = 2.0
        return (nambu_periodized/Nc)


    def periodize_nambu(self, kx: float, ky: float, ii: int): # Green periodization
        """ """
        nambu_ktilde = self.build_gf_ktilde(kx, ky, ii)

        return self.periodize(kx, ky, nambu_ktilde) 


    def stiffness(self, kx: float, ky: float, ii: int) -> float:
        """ """
        nambu_periodized = self.periodize(kx, ky, self.build_gf_ktilde(kx, ky, ii)) #self.periodize_nambu(kx, ky, ii)
        coskx: float = np.cos(kx) 
        cosky: float = np.cos(ky)
        tperp = -(coskx - cosky)*(coskx - cosky) # t_perp = -1.0
        tperp_squared = 2.0*tperp*tperp # integrated over kz (integrate cos(kz)**2.0 = 2.0)
        return (-1.0 * np.real(-tperp_squared*
                                (2.0*nambu_periodized[0, 2]*nambu_periodized[2, 0] +
                                 4.0*(nambu_periodized[0, 3]*nambu_periodized[3, 0]) +
                                 2.0*nambu_periodized[1, 3]*nambu_periodized[3, 1]
                                )

                             )
                )

    def stiffness_orbital(self, kx, ky, ii):
        nambu_periodized = self.periodize_orbitale(kx, ky, self.build_gf_ktilde(kx, ky, ii)) #self.periodize_nambu(kx, ky, ii)
        coskx: float = np.cos(kx) 
        cosky: float = np.cos(ky)
        tperp = -(coskx - cosky)*(coskx - cosky) # t_perp = -1.0
        tperp_squared = 2.0*tperp*tperp # integrated over kz (integrate cos(kz)**2.0 = 2.0)
        return (-1.0 * np.real(-tperp_squared*
                                4.0*(nambu_periodized[0, 3]*nambu_periodized[2, 1] + nambu_periodized[1, 2]*nambu_periodized[3, 0]) 
                              )
                )


    def eps0(self, kx, ky):
        return (-2.0*self.t*(np.cos(kx) + np.cos(ky)) - 2.0*self.tp*np.cos(kx + ky) )

    def periodize_cumulant(self, kx: float, ky: float, ii: int): # cumulant periodization
        """ """
        tmp = linalg.inv(self.periodize(kx, ky, self.cumulants[ii]))
        
        tmp[0, 0] -= self.eps0(kx, ky); tmp[1, 1] -= self.eps0(kx+np.pi, ky+np.pi)
        tmp[2, 2] += self.eps0(kx, ky); tmp[3, 3] += self.eps0(kx+np.pi, ky+np.pi)
        
        return linalg.inv(tmp)


    def stiffness_cum(self, kx: float, ky: float, ii: int) -> float:
        """ """
        
        nambu_periodized = self.periodize_cumulant(kx, ky, ii) #linalg.inv(tmp.copy())
        coskx: float = np.cos(kx) 
        cosky: float = np.cos(ky)
        tperp = -(coskx - cosky)*(coskx - cosky) # t_perp = -1.0
        tperp_squared = 2.0*tperp*tperp # integrated over kz (integrate cos(kz)**2.0 = 2.0)
        return (-1.0 * np.real(-tperp_squared*
                                (2.0*nambu_periodized[0, 2]*nambu_periodized[2, 0] +
                                 4.0*nambu_periodized[0, 3]*nambu_periodized[3, 0] +
                                 2.0*nambu_periodized[1, 3]*nambu_periodized[3, 1]
                                )

                             )
                )   


    # for d_z p-wave SC, nambu space is not enlarged.
    def stiffness_trace(self, kx: float, ky: float, ii: int) -> float:
        """4/N_c Trace(F F^Dag) """
        gf_ktilde = self.build_gf_ktilde(kx, ky, ii)
        trace = np.trace(np.dot(gf_ktilde[:4:, 4::], gf_ktilde[4::, :4:]))
        coskx: float = np.cos(kx) 
        cosky: float = np.cos(ky)
        tperp = -(coskx - cosky)*(coskx - cosky) # t_perp = -1.0
        tperp_squared = 2.0*tperp*tperp # integrated over kz (integrate cos(kz)**2.0 = 2.0)    
        return (tperp_squared*np.real(trace))

