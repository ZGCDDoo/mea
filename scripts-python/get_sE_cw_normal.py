from mea.tools import kramerskronig as kk
import json
from scipy import linalg
import numpy as np
from mea.model import green, periodize
from mea import acon
from scipy.interpolate import interp1d
import sys

with open("statsparams0.json") as fin:
    mu = json.load(fin)["mu"][0]


Aw_tir = np.loadtxt("Aw_t.dat")
w_vec = Aw_tir[:, 0]
Aw_tir = np.delete(Aw_tir, 0, axis=1).copy()


gfvec_aux_irtw = np.zeros(Aw_tir.shape, dtype=complex)
for i in range(Aw_tir.shape[1]):
    gfvec_aux_irtw[:, i] = -0.5*(kk.KramersKroning(w_vec, Aw_tir[:, i]) + 1.0j*Aw_tir[:, i])

gfvec_aux_irw = green.t_to_ir(w_vec, gfvec_aux_irtw)
gfvec_aux_cw = green.ir_to_c(gfvec_aux_irw)

sEvec_cw = np.zeros(gfvec_aux_cw.shape, dtype=complex)

for (j, gf) in enumerate(gfvec_aux_cw):
    sEvec_cw[j] = -linalg.inv(gf) + np.eye(4)*(w_vec[j] + mu)


green.save_gf_c("self_ctow_new.dat", w_vec, sEvec_cw)



