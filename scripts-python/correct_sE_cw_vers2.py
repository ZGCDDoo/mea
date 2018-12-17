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

tmp_green = np.loadtxt(str(sys.argv[1]))

ac = acon.ACon(Aw_tir + 0.0j, w_vec)
ac.build_gf_aux_t(fout_gf_aux_t="Aw_aux_to_acon.dat")
Aw_aux_tir = ac.gf_aux_t.real

queue_low = np.linspace(-80, -10, 1000, endpoint=False)
queue_high = np.linspace(10, 80, 1000)
grid_medium = np.linspace(-10, 10, 10000, endpoint=False)

newgrid = np.concatenate((queue_low, grid_medium, queue_high))

new_Aw_aux_tir = np.zeros((newgrid.shape[0], Aw_aux_tir.shape[1]))

for ii in range(Aw_aux_tir.shape[1]):
    fqueue = interp1d(w_vec, Aw_aux_tir[:, ii], kind='linear')
    fmedium = interp1d(w_vec, Aw_aux_tir[:, ii], kind="cubic")
    new_Aw_aux_tir[:, ii] = np.concatenate((fqueue(queue_low), fmedium(grid_medium), fqueue(queue_high) ))        

fqueue = interp1d(tmp_green[:,0], tmp_green[:,1], kind='linear')
fmedium = interp1d(tmp_green[:,0], tmp_green[:,1], kind="cubic")
new_Aw_aux_tir[:, int(sys.argv[2])] = np.concatenate((fqueue(queue_low), fmedium(grid_medium), fqueue(queue_high) ))        


new_Aw_tir = ac.build_Aw_t(new_Aw_aux_tir)
np.savetxt("newaw.dat", np.concatenate((newgrid[:, np.newaxis], new_Aw_tir), axis=1) )

gfvec_aux_irtw = np.zeros(new_Aw_tir.shape, dtype=complex)
for i in range(new_Aw_tir.shape[1]):
    gfvec_aux_irtw[:, i] = -0.5*(kk.KramersKroning(newgrid, new_Aw_tir[:, i]) + 1.0j*new_Aw_tir[:, i])

gfvec_aux_irw = green.t_to_ir(newgrid, gfvec_aux_irtw)
gfvec_aux_cw = green.ir_to_c(gfvec_aux_irw)

sEvec_cw = np.zeros(gfvec_aux_cw.shape, dtype=complex)

for (j, gf) in enumerate(gfvec_aux_cw):
    sEvec_cw[j] = -linalg.inv(gf) + np.eye(4)*(newgrid[j] + mu)


green.save_gf_c("self_ctow_new.dat", newgrid, sEvec_cw)

