# coding: utf-8
import numpy as  np
from mea import acon
import shutil
from mea.tools import kramerskronig as kk
from copy import deepcopy
import json

#copyright Char
with open("statsparams0.json") as fin:
    params_fin = json.load(fin)
    mu = params_fin["mu"][0]
    beta = params_fin["beta"][0]
    t = 1.0
    tp = params_fin["tp"][0]
    #tpp = params_fin["tpp"][0]

with open("hyb_moy.json") as fin:
    params_fin = json.load(fin)
    hybvec = np.array(params_fin["0Pi"]["real"]) + 1.0j*np.array(params_fin["0Pi"]["imag"] )

with open("self_moy.json") as fin:
    params_fin = json.load(fin)
    selfvec = np.array(params_fin["0Pi"]["real"]) + 1.0j*np.array(params_fin["0Pi"]["imag"] )

znvec = np.array([(2.0*n + 1.0)*np.pi/beta for n in selfvec.shape[0]])


#AC of hyb and g_cl
achyb = acon.ACon(np.transpose([hybvec.real, hyvec.imag]), znvec) ; achyb.run_acon()
shutil.move("Result_OME", "Result_OME_hyb")


gfaux = np.zeros(hybvec.shape[0], dtype=complex)
gfaux = 1.0/(1.0j*znvec - hybvec - selfvec)

acgf = acon.ACon(np.transpose([gfaux.real, gfaux.imag]), znvec); acgf.run_acon()
shutil.move("Result_OME", "Result_OME_gfaux")

#now get everything on real axis by kramers-kronig


Aw_t_list_hyb = deepcopy(achyb.Aw_t_list)
Aw_t_list_gf = deepcopy(acgf.Aw_t_list)
hybvec_w_list = []; hybvec_w_list = []
gfvec_w_list = []; gfvec_w_list = [] 
w_vec_list = []


for (ii, (w_vec, Aw_t)) in enumerate(zip(achyb.w_vec_list, achyb.Aw_t_list)):
    if w_vec is None:
        continue
    hybvec_w = np.zeros(Aw_t.shape, dtype=complex)
    hybvec_w = -0.5*(kk.KramersKroning(w_vec, Aw_t) + 1.0j*Aw_t)
    hybvec_w_list.append(hybvec_w)
    w_vec_list.append(w_vec)    


for (ii, (w_vec, Aw_t)) in enumerate(zip(acgf.w_vec_list, acgf.Aw_t_list)):
    if w_vec is None:
        continue
    gfvec_w = np.zeros(Aw_t.shape, dtype=complex)
    gfvec_w = -0.5*(kk.KramersKroning(w_vec, Aw_t) + 1.0j*Aw_t)
    gfvec_w_list.append(gfvec_w)


for (n, ((gfvec_w, w_vec)) in enumerate(zip(gfvec_w_list, w_vec_list)):
    gfaux = 1.0/(1.0j*znvec - hybvec - selfvec)
    selfvec_w = -1.0/gf + w_vec + hybvec_w
    fname = "self_w" + str(n) + ".dat"
    np.savetxt(fname, np.transpose([w_vec, selfvec_w.real, selfvec_w.imag]))

