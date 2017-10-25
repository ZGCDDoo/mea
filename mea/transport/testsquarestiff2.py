# coding: utf-8
from stiffsquare import nambu_square_coexistence_vers2 as periodize_nambu
from stiffsquare import stiffness_square
import numpy as np


#=========== DEBUT de definition de la self-fictive, juste pour demo, PAS IMPORTANT ======================
beta = 40.0
znvec_tmp = np.array([(2.0*n + 1.0)*np.pi/beta for n in range(20)], dtype=complex)
sEvec_c_tmp = np.zeros((20, 8, 8), dtype=complex)

MM = 0.0517
pp = 0.005
dd = 0.03

for (ii, zz) in enumerate(znvec_tmp):
    # Gup
    sEvec_c_tmp[ii, 0, 0] = sEvec_c_tmp[ii, 3, 3] = -MM
    sEvec_c_tmp[ii, 1, 1] = sEvec_c_tmp[ii, 2, 2] = MM  
    
    # Gdown*
    sEvec_c_tmp[ii, 4, 4] = sEvec_c_tmp[ii, 6, 6] = -MM
    sEvec_c_tmp[ii, 5, 5] = sEvec_c_tmp[ii, 7, 7] = MM 

    #F
    FF = 1.0/(zz*zz)*np.array([
                  [0.0, dd + pp, -dd + pp, 0.0],
                  [dd - pp, 0.0, 0.0, -dd + pp],
                  [-dd - pp, 0.0, 0.0, dd + pp],
                  [0.0, -dd - pp, dd - pp, 0.0]
                   ], dtype=complex)

    sEvec_c_tmp[ii, :4:, 4::] = FF
    
    #FDag
    sEvec_c_tmp[ii, 4::, :4:] = np.conjugate(np.transpose(FF))


#=========== DEBUT de definition de la self-fictive, juste pour demo ======================

(znvec, sEvec_c) = (znvec_tmp, sEvec_c_tmp) # (qcm.sE_cluster ou la petite loop que tu fais pour la discretizer comme en haut)
modelSC = periodize_nambu.ModelNambu(t=1.0, tp=0.40, tpp=0.0, mu=2.9246671954980012, z_vec=1.0j*znvec, sEvec_c=sEvec_c) # tu dois specifier que znvec est complexe
stiffness_square.stiffness(modelSC)
