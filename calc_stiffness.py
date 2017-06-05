from mea.model import periodize_nambu, nambu
import numpy as np
from scipy.integrate import dblquad
import json

(zn_vec, sEvec_c) = nambu.read_nambu_c("self_moy.dat")

with open("statsparams0.json") as fin:
    params = json.load(fin)

mu = params["mu"][0]
beta = params["beta"][0]
tp = params["tp"][0]
stiffness = 0.0


model_sc = periodize_nambu.ModelNambu(1.0, tp, mu, 1.0j*zn_vec, sEvec_c) 
Y1 = model_sc.Y1Limit
Y2 = model_sc.Y2Limit

#print(model_sc.periodize_nambu(-3.14, 0, 0)[0, 0])
#print(model_sc.periodize_nambu(-3.14, 0, 0)[0, 1])
#print(model_sc.periodize_nambu(-3.14, 0, 0)[1, 0])
#print(model_sc.periodize_nambu(-3.14, 0, 0)[1, 1])

for ii in range(zn_vec.shape[0]):
    stiffness += 2.0/beta*1.0/(2.0*np.pi)**2*dblquad(model_sc.stiffness, -np.pi, np.pi, Y1, Y2, args=(ii,) )[0]
    print("stiffness = ", stiffness)

print("\n\n\n lattice stiffness \n\n.")
print("stiffness = ", stiffness)

np.savetxt("stiffness.dat", np.array([stiffness]))

