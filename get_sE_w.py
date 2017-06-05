from mea.tools import kramerskronig as kk
import json
from scipy import linalg
import nmpy as np
from mea.model import nambu


with open("statsparams0.json") as fin:
    mu = json.load(fin)["mu"][0]


Aw_t = np.loadtxt("Aw_t.dat")
w_vec = Aw_t[:, 0]
Aw_t = np.delete(Aw_t, 0, axis=1).copy()


gfvec_aux_irtw = np.zeros(Aw_t.shape, dtype=complex)
for i in range(Aw_t.shape[1]):
    gfvec_aux_irtw[:, i] = -0.5*(kk.KramersKroning(w_vec, Aw_t[:, i]) + 1.0j*Aw_t[:, i])

gfvec_aux_irw = nambu.t_to_irw(w_vec, gfvec_aux_irtw)

sEvec_irw = np.zeros(gfvec_aux_irw.shape, dtype=complex)

for (j, gf) in enumerate(gfvec_aux_irw):
    sEvec_irw[j] -= linalg.inv(gf)
    sEvec_irw[j, :4:, :4:] += (w_vec[j] + mu)
    sEvec_irw[j, 4::, 4::] -= (-w_vec[j] + mu)


nambu.save_nambu_c("self_ctow.dat", w_vec, nambu.ir_to_c(sEvec_irw))

