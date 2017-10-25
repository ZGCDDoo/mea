# coding: utf-8
from stiffsquare import nambu_square_coexistence_vers2 as periodize_nambu
from stiffsquare import stiffness_square
import numpy as np


#=========== DEBUT de definition de la self-fictive, juste pour demo, PAS IMPORTANT ======================
beta = 40.0
znvec_tmp = np.array([(2.0*n + 1.0)*np.pi/beta for n in range(2)], dtype=complex)
sEvec_c_tmp = np.zeros((2, 8, 8), dtype=complex)

for (ii, zz) in enumerate(znvec_tmp):
    sEvec_c_tmp[ii, :4:, :4:] = (0.1* + 1.0/(1.0j*zz))*np.eye(4) 
    sEvec_c_tmp[ii, 4::, 4::] = (0.1* + 1.0/(1.0j*zz))*np.eye(4) 
    sEvec_c_tmp[ii, :4:, 4::] = 0.2/(zz*zz)*np.eye(4)
    sEvec_c_tmp[ii, 4::, :4:] = 0.2/(zz*zz)*np.eye(4)


#=========== DEBUT de definition de la self-fictive, juste pour demo ======================

(znvec, sEvec_c) = (znvec_tmp, sEvec_c_tmp) # (qcm.sE_cluster ou la petite loop que tu fais pour la discretizer comme en haut)
modelSC = periodize_nambu.ModelNambu(t=1.0, tp=0.40, tpp=0.0, mu=2.9246671954980012, z_vec=1.0j*znvec, sEvec_c=sEvec_c) # tu dois specifier que znvec est complexe
stiffness_square.stiffness(modelSC)
