# coding: utf-8
import numpy as  np
from mea import acon
from mea.model import green
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



(znvec2, gfvec_c) = green.read_green_c("green_moy.dat")
(znvec, hybvec_c) = green.read_green_c("hyb_moy.dat")
hybvec_ir = green.c_to_ir(hybvec_c) ; gfvec_ir = green.c_to_ir(gfvec_c)
np.testing.assert_allclose(znvec, znvec2)

#AC of hyb and g_cl
achyb = acon.ACon(green.ir_to_t(znvec, hybvec_ir), znvec) ; achyb.acon()
shutil.move("Result_OME", "Result_OME_hyb")

acgf = acon.ACon(green.ir_to_t(znvec, gfvec_ir), znvec); acgf.acon()
shutil.move("Result_OME", "Result_OME_gf")

#now get everything on real axis by kramers-kronig


Aw_t_list_hyb = deepcopy(achyb.Aw_t_list)
Aw_t_list_gf = deepcopy(acgf.Aw_t_list)
hybvec_irtw_list = []; hybvec_irw_list = []
gfvec_irtw_list = []; gfvec_irw_list = [] 
w_vec_list = []


for (ii, (w_vec, Aw_t)) in enumerate(zip(achyb.w_vec_list, achyb.Aw_t_list)):
    if w_vec is None:
        continue
    hybvec_irtw = np.zeros(Aw_t.shape, dtype=complex)
    for jj in range(Aw_t.shape[1]):
        hybvec_irtw[:, jj] = -0.5*(kk.KramersKroning(w_vec, Aw_t[:, jj]) + 1.0j*Aw_t[:, jj])
    hybvec_irtw_list.append(hybvec_irtw)
    w_vec_list.append(w_vec)    
    hybvec_irw_list.append(green.t_to_ir(w_vec, hybvec_irtw))



for (ii, (w_vec, Aw_t)) in enumerate(zip(acgf.w_vec_list, acgf.Aw_t_list)):
    if w_vec is None:
        continue
    gfvec_irtw = np.zeros(Aw_t.shape, dtype=complex)
    for jj in range(Aw_t.shape[1]):
        gfvec_irtw[:, jj] = -0.5*(kk.KramersKroning(w_vec, Aw_t[:, jj]) + 1.0j*Aw_t[:, jj])
    gfvec_irtw_list.append(gfvec_irtw)
    gfvec_irw_list.append(green.t_to_ir(w_vec, gfvec_irtw))



#let us save the hyb and gf
for (ii, (w_vec, hybvec, gfvec)) in enumerate(zip(w_vec_list, hybvec_irw_list, gfvec_irw_list)):
    fout_hyb = "hyb_irtow" + str(ii) + ".dat"
    fout_gf = "gf_irtow" + str(ii) + ".dat"
    print("in save fct.....\n\n")
    green.save_gf_ir(fout_hyb, w_vec, hybvec)
    green.save_gf_ir(fout_gf, w_vec, gfvec)


#now extract self-energy
hybvec_cw_list = map(green.ir_to_c, hybvec_irw_list)
gfvec_cw_list = map(green.ir_to_c, gfvec_irw_list)
sEvec_cw_list = []


for(ii, (w_vec, hybvec_cw, gfvec_cw)) in enumerate(zip(w_vec_list, hybvec_cw_list, gfvec_cw_list)):
    sEvec_cw = np.zeros(hybvec_cw.shape, dtype=complex)

    for (j, (hyb, gf)) in enumerate(zip(hybvec_cw, gfvec_cw)):
        ww = (w_vec[j] + mu)
        sEvec_cw[j] = -linalg.inv(gf) + (ww*np.eye(4) - tloc -hyb)

    sEvec_cw_list.append(sEvec_cw)
    fout_sE =  "self_ctow" + str(ii) + ".dat"

    green.save_gf_c(fout_sE, w_vec, sEvec_cw)

