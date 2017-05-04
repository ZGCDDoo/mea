#calc_dos_fermi.py

from ..model import green, periodize
import sys, os
import json

fname = str(sys.argv[1])

with open("statsparams.dat") as fin:
    mu = json.load(fin)["mu"]

(w_vec, sEvec_cw) = green.read_green_c(fname)

model = periodize.Model(1.0, 0.4, mu, w_vec, sEvec_cw)
periodize.calc_dos(model, "dos_calcdos.dat")
periodize.fermi_surface(model, 0.0, "fermi_surface.dat")