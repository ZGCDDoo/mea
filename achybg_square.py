# coding: utf-8

import numpy as  np
from mea import acon_square as acon
import shutil
from mea.tools import kramerskronig as kk
from copy import deepcopy
import json
import sys

iter_start = int(sys.argv[1])
iter_end = int(sys.argv[2])

with open("statsparams0.json") as fin:
    params_fin = json.load(fin)
    mu = params_fin["mu"][0]
    beta = params_fin["beta"][0]
    t = 1.0
    tp = params_fin["tp"][0]
    #tpp = params_fin["tpp"][0]


fname_hyb = "Hyb" + str(iter_start) + ".json"

with open(fname_hyb) as fin:
    params_fin = json.load(fin)
    hybvec = np.array(params_fin["0Pi"]["real"], dtype=complex) + 1.0j*np.array(params_fin["0Pi"]["imag"], dtype=complex )

fname_self = "Self" + str(iter_start) + ".json"
with open(fname_self) as fin:
    params_fin = json.load(fin)
    selfvec = np.array(params_fin["0Pi"]["real"], dtype=complex) + 1.0j*np.array(params_fin["0Pi"]["imag"], dtype=complex)

for ii in range(iter_start + 1, iter_end + 1):
    fname_hyb = "Hyb" + str(ii) + ".json"

    with open(fname_hyb) as fin:
        params_fin = json.load(fin)
        hybvec += np.array(params_fin["0Pi"]["real"]) + 1.0j*np.array(params_fin["0Pi"]["imag"] )

    fname_self = "Self" + str(ii) + ".json"
    with open(fname_self) as fin:
        params_fin = json.load(fin)
        selfvec += np.array(params_fin["0Pi"]["real"], dtype=complex) + 1.0j*np.array(params_fin["0Pi"]["imag"], dtype=complex)

NN = iter_end - iter_start + 1
selfvec /= NN
hybvec /= NN

znvec = np.array([(2.0*n + 1.0)*np.pi/beta for n in range(selfvec.shape[0])])

#AC of hyb and g_cl
achyb = acon.ACon(hybvec[:, np.newaxis], znvec) ; achyb.run_acon()
shutil.move("Result_OME", "Result_OME_hyb")

gfaux = 1.0/(1.0j*znvec + mu - hybvec - selfvec)


acgf = acon.ACon(gfaux[:, np.newaxis], znvec); acgf.run_acon()
shutil.move("Result_OME", "Result_OME_gfaux")


Aw_t_list_hyb = deepcopy(achyb.Aw_t_list)
Aw_t_list_gf = deepcopy(acgf.Aw_t_list)
hybvec_w_list = []; hybvec_w_list = []
gfvec_w_list = []; gfvec_w_list = [] 
w_vec_list = []


for (ii, (w_vec, Aw_t)) in enumerate(zip(achyb.w_vec_list, achyb.Aw_t_list)):
    if w_vec is None:
        continue
    hybvec_w = np.zeros(Aw_t.shape, dtype=complex)
    hybvec_w = -0.5*(kk.KramersKroning(w_vec, Aw_t[:,0]) + 1.0j*Aw_t[:,0])
    hybvec_w_list.append(hybvec_w)
    w_vec_list.append(w_vec)    


for (ii, (w_vec, Aw_t)) in enumerate(zip(acgf.w_vec_list, acgf.Aw_t_list)):
    if w_vec is None:
        continue
    gfvec_w = np.zeros(Aw_t.shape, dtype=complex)
    gfvec_w = -0.5*(kk.KramersKroning(w_vec, Aw_t[:,0]) + 1.0j*Aw_t[:,0])
    gfvec_w_list.append(gfvec_w)


for (n, (gfvec_w, w_vec)) in enumerate(zip(gfvec_w_list, w_vec_list)):
    selfvec_w = -1.0/gfvec_w + w_vec + mu + hybvec_w
    fname = "self_w" + str(n) + ".dat"
    np.savetxt(fname, np.transpose([w_vec, selfvec_w.real, selfvec_w.imag]))

