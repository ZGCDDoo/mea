import os, sys, inspect
import numpy as np
import scipy.integrate
import json

#filedir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(filedir)
#sys.path.insert(0, parentdir)
#sys.path.insert(0, filedir)


from mea import auxfn
from mea.model import green, periodize
from mea.transport import sigmadc
from scipy.integrate import simps

cwd = os.getcwd()



with open("statsparams0.json") as fin:
    params = json.load(fin)
    mu = params["mu"][0]
    beta = params["beta"][0]


gf_aux = auxfn.GFAux(fin_sE_to=os.path.join(cwd, "self_moy.dat"), mu=mu, delta=0.001)
gf_aux.ac(fin_OME_default=os.path.join(cwd, "OME_default.dat"), fin_OME_input=os.path.join(cwd, "OME_input.dat"))
gf_aux.get_sEvec_w_list()

sEvec_cw_list = list(map(green.ir_to_c, gf_aux.sEvec_irw_list))
w_vec_list = gf_aux.w_vec_list


for (i, (sEvec_cw, w_vec)) in enumerate(zip(sEvec_cw_list, w_vec_list)):

    model = periodize.Model(1.0, 0.4, mu, w_vec, sEvec_cw)
    fout_name = "dos" + str(i) + ".dat"
    fout_name_dos_trace = "dos_trace" + str(i) + ".txt"
    dos = periodize.calc_dos(model, fout_name)
    dos_trace = periodize.calc_dos_with_trace(model, fout_name_dos_trace)
    dos_fct = np.loadtxt(fout_name)
    dos_trace = np.loadtxt(fout_name_dos_trace)
    print("dos normalisation = ", simps(dos_fct[:, 1], dos_fct[:, 0])/(2.0*np.pi))
    print("dos_trace normalisation = ", simps(dos_trace[:, 1], dos_trace[:, 0])/(2.0*np.pi))
    print("zkweight = ", periodize.zk_weight(model, np.pi, np.pi) )
    fout_fermi = "fermi_surface" + str(i) + ".dat"
    periodize.fermi_surface(model, w_value=0.0, fout=fout_fermi)
    sdc = sigmadc.SigmaDC(w_vec, sEvec_cw, beta=beta, mu=mu)
    sdc.calc_sigmadc()