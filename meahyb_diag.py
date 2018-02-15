# coding: utf-8
import numpy as np
from mea import acon
import shutil
from mea.tools import kramerskronig as kk
from copy import deepcopy
import json
from scipy import linalg
import sys


mu = float(sys.argv[1])

sEvec_to = np.loadtxt("self_moy.dat")
hybvec_to = np.loadtxt("hyb_moy.dat")
znvec = np.copy(sEvec_to[:, 0])

sEvec = 0.5 * (sEvec_to[:, 1] + 1.0j * sEvec_to[:, 2]) + \
    0.5 * (sEvec_to[:, 3] + 1.0j * sEvec_to[:, 4])
hybvec = 0.5 * (hybvec_to[:, 1] + 1.0j * hybvec_to[:, 2]) + \
    0.5 * (hybvec_to[:, 3] + 1.0j * hybvec_to[:, 4])
gfvec_aux = np.zeros(sEvec.shape, dtype=complex)

for ii in range(sEvec.shape[0]):
    gfvec_aux[ii] = 1.0 / ((1.0j * znvec[ii] + mu) - hybvec[ii] - sEvec[ii])


# AC of hyb and gfaux
ac_hyb = acon.ACon(hybvec[:][:, np.newaxis], znvec, loc=(
    0,), non_loc=(), non_loc_sign=(), non_loc_to_conjugate=())
ac_hyb.run_acon()
shutil.move("Result_OME", "Result_OME_hyb")

ac_aux = acon.ACon(gfvec_aux[:][:, np.newaxis], znvec, loc=(
    0,), non_loc=(), non_loc_sign=(), non_loc_to_conjugate=())
ac_aux.run_acon()
shutil.move("Result_OME", "Result_OME_gfaux")


# now get everything on real axis by kramers-kronig


gfvec_aux_w_list = []
hybvec_w_list = []


for (ii, (w_vec, Aw_t)) in enumerate(zip(ac_hyb.w_vec_list, ac_hyb.Aw_t_list)):
    if w_vec is None:
        continue
    hybvec_w = np.zeros(Aw_t.shape, dtype=complex)
    for jj in range(Aw_t.shape[1]):
        hybvec_w[:, jj] = -0.5 * \
            (kk.KramersKroning(w_vec, Aw_t[:, jj]) + 1.0j * Aw_t[:, jj])
    hybvec_w_list.append(hybvec_w)


for (ii, (w_vec, Aw_t)) in enumerate(zip(ac_aux.w_vec_list, ac_aux.Aw_t_list)):
    if w_vec is None:
        continue
    gfvec_aux_w = np.zeros(Aw_t.shape, dtype=complex)
    for jj in range(Aw_t.shape[1]):
        gfvec_aux_w[:, jj] = -0.5 * \
            (kk.KramersKroning(w_vec, Aw_t[:, jj]) + 1.0j * Aw_t[:, jj])
    gfvec_aux_w_list.append(gfvec_aux_w)


# now extract self-energy
sEvec_w_list = []


for(ii, (w_vec, gfvec_aux_w)) in enumerate(zip(ac_aux.w_vec_list, gfvec_aux_w_list)):
    sEvec_w = np.zeros(gfvec_aux_w.shape, dtype=complex)
    hybvec_w = hybvec_w_list[ii]

    if hybvec_w is None or gfvec_aux_w is None:
        continue

    for (jj, gf) in enumerate(gfvec_aux_w):
        ww = (w_vec[jj] + mu)
        hyb = hybvec_w[jj]
        sEvec_w[jj] = -1.0 / gf + (ww - hyb)

    sEvec_w_list.append(sEvec_w)
    fout_sE = "self_tow" + str(ii) + ".dat"

    print("shapes = ", w_vec.shape, " ", sEvec_w.shape)
    np.savetxt(fout_sE, np.transpose(
        [w_vec, sEvec_w[:, 0].real, sEvec_w[:, 0].imag]))
