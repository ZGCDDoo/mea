# coding: utf-8
import numpy as  np
from mea import acon
from mea.model.io_triangle import IOTriangle as green
import shutil
from mea.tools import kramerskronig as kk
from copy import deepcopy
import json
from scipy import linalg
from collections import namedtuple

#change t, mu, and tp for a namedtuple with name ...?


with open("statsparams0.json") as fin:
    params_fin = json.load(fin)
    mu = params_fin["mu"][0]
    t = 1.0
    tp = params_fin["tp"][0]

tloc = np.array([[0.0, -t, -tp, -t],
                [-t, 0.0, -t, 0.0],
                [-tp, -t, 0.0, -t],
                [-t, 0.0, -t, 0.0]])



(znvec, sEvec_c) = green().read_green_c("self_moy.dat")
(znvec2, hybvec_c) = green().read_green_c("hyb_moy.dat")

gfvec_aux_c = np.zeros(sEvec_c.shape, dtype=complex)


for ii in range(sEvec_c.shape[0]):
    hyb = hybvec_c[ii, 0, 0]
    gfvec_aux_c[ii] = linalg.inv((1.0j*znvec[ii] + mu)*np.eye(4) - hyb - tloc - sEvec_c[ii])

gfvec_aux_ir = green().c_to_ir(gfvec_aux_c)

#AC of hyb and aux
ac_hyb = acon.ACon(hybvec_c[:, 0, 0][:,np.newaxis], znvec, loc=(0,), non_loc=(), non_loc_sign=(), non_loc_to_conjugate=() ) ; ac_hyb.acon()
shutil.move("Result_OME", "Result_OME_hyb")

ac_aux = acon.ACon(green().ir_to_t(znvec, gfvec_aux_ir), znvec) ; ac_aux.run_acon()
shutil.move("Result_OME", "Result_OME_gfaux")


#now get everything on real axis by kramers-kronig


gfvec_aux_irtw_list = []; gfvec_aux_irw_list = [] 
hybvec_cw_list=[]


for (ii, (w_vec, Aw_t)) in enumerate(zip(ac_hyb.w_vec_list, ac_hyb.Aw_t_list)):
    if w_vec is None:
        continue
    hybvec_cw = np.zeros(Aw_t.shape, dtype=complex)
    for jj in range(Aw_t.shape[1]):
        hybvec_cw[:, jj] = -0.5*(kk.KramersKroning(w_vec, Aw_t[:, jj]) + 1.0j*Aw_t[:, jj])
    hybvec_cw_list.append(hybvec_cw)


for (ii, (w_vec, Aw_t)) in enumerate(zip(ac_aux.w_vec_list, ac_aux.Aw_t_list)):
    if w_vec is None:
        continue
    gfvec_aux_irtw = np.zeros(Aw_t.shape, dtype=complex)
    for jj in range(Aw_t.shape[1]):
        gfvec_aux_irtw[:, jj] = -0.5*(kk.KramersKroning(w_vec, Aw_t[:, jj]) + 1.0j*Aw_t[:, jj])
    gfvec_aux_irtw_list.append(gfvec_aux_irtw)
    gfvec_aux_irw_list.append(green().t_to_ir(w_vec, gfvec_aux_irtw))


#now extract self-energy
gfvec_aux_cw_list = map(green().ir_to_c, gfvec_aux_irw_list)
sEvec_cw_list = []


for(ii, (w_vec, gfvec_aux_cw)) in enumerate(zip(ac_aux.w_vec_list, gfvec_aux_cw_list)):
    sEvec_cw = np.zeros(gfvec_aux_cw.shape, dtype=complex)
    hybvec_cw = hybvec_cw_list[ii]

    for (jj, gf) in enumerate(gfvec_aux_cw):
        ww = (w_vec[jj] + mu)
        hyb = hybvec_cw[jj]
        sEvec_cw[jj] = -linalg.inv(gf) + (ww*np.eye(4) - tloc - hyb)

    sEvec_cw_list.append(sEvec_cw)
    fout_sE =  "self_ctow" + str(ii) + ".dat"

    green().save_gf_c(fout_sE, w_vec, sEvec_cw)
