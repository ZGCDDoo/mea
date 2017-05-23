# coding: utf-8
#testing the fact that the anormal self-energy should be real, even on the cluster.
#in fact, i show here, that the periodized green function has off-diagonal blocks equal (the real part) only if the
# anormal self-energy is chosen real!!!


from mea.model import nambu
from mea.model import periodize_nambu
import numpy as np

(znvec, sEvec_c) = nambu.read_nambu_c("self_moy.dat")
import json
with open("statsparams0.json") as fin:
    mu=json.load(fin)["mu"][0]

rr = np.random.uniform(low=-0.01, high=0.01, size=2)
nn = np.random.normal(size=2)
kx = np.pi + rr[0]
ky = 0.0 +  rr[1]
t = nn[0]
tp = nn[1] 

per_gf = periodize_nambu.periodize_nambu(kx, ky, 1.0, 0.4, sEvec_c[0], 1.0j*znvec[0], mu)
gf_ktilde = periodize_nambu.build_gf_ktilde(kx, ky ,1.0 ,0.4, sEvec_c[0], 1.0j*znvec[0], mu)
print("Periodize gf = \n", per_gf)
