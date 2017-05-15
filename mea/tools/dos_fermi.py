# calc_dos_fermi.py

from ..model import green
from ..model import periodize_class as perc
import json


def calc_dos_fermi(fname):
    """ """

    with open("statsparams0.json") as fin:
        mu = json.load(fin)["mu"][0]

    (w_vec, sEvec_cw) = green.read_green_c(fname)

    model = perc.Model(1.0, 0.4, mu, w_vec, sEvec_cw)
    model.calc_dos("dos_calcdos.dat")
    #model.fermi_surface(0.0, "fermi_surface.dat")