# coding: utf-8

#same as achybg.py, but if stopped before, can restart from the output files
#of the analytic contiunation.

import numpy as  np
from mea import acon
from mea.model import green
import shutil
from mea.tools import kramerskronig as kk
from copy import deepcopy
import json
from scipy import linalg

t=1.0; tp=0.4
tloc = np.array([[0.0, -t, -tp, -t],
                [-t, 0.0, -t, 0.0],
                [-tp, -t, 0.0, -t],
                [-t, 0.0, -t, 0.0]])


with open("statsparams0.json") as fin:
    mu = json.load(fin)["mu"][0]
 


(w_vec2, gfvec_irw) = green.read_green_ir("gf_irtow0.dat")
(w_vec, hybvec_irw) = green.read_green_ir("hyb_irtow0.dat")
(hybvec_cw, gfvec_cw) = map(green.ir_to_c, [hybvec_irw, gfvec_irw])







sEvec_cw = np.zeros(hybvec_cw.shape, dtype=complex)
for (j, (hyb, gf)) in enumerate(zip(hybvec_cw, gfvec_cw)):
    ww = (w_vec[j] + mu)
    sEvec_cw[j] = -linalg.inv(gf) + (ww*np.eye(4) - tloc -hyb)

ii=0
fout_sE =  "self_ctow" + str(ii) + ".dat"

green.save_gf_c(fout_sE, w_vec, sEvec_cw)

