import periodize_nambu
import nambu
import numpy as np
from scipy.integrate import dblquad
import json

(zn_vec, sEvec_c) = nambu.read_nambu_c("self_moy.dat")

with open("statsparams0.json") as fin:
    params = json.load(fin)

mu = params["mu"][0]
beta = params["beta"][0]
stiffness = 0.0


Y1 = periodize_nambu.Y1Limit
Y2 = periodize_nambu.Y2Limit

for (zn, sE) in zip(zn_vec, sEvec_c):
    stiffness += 2.0/beta*1.0/(2.0*np.pi)**2*dblquad(periodize_nambu.stiffness, -np.pi, np.pi, Y1, Y2, args=(1.0, 0.4, sE, 1.0j*zn, mu) )[0]
    print("stiffness = ", stiffness)

print("\n\n\n lattice stiffness \n\n.")
print("stiffness = ", stiffness)

np.savetxt("stiffness.dat", np.array([stiffness]))


