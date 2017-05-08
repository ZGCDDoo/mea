from mea.model import green
from mea.transport import sigmadc
import os, json


with open("statsparams0.json") as fin:
    params = json.load(fin)
    mu = params["mu"][0]
    beta = params["beta"][0]

for i in range(10):
    
    fname: str = "self_ctow" + str(i) + ".dat"
    if not os.path.isfile(fname):
        break

    (wvec, sEvec_c) = green.read_green_c(fname)
    SDC = sigmadc.SigmaDC(wvec, sEvec_c, beta=beta, mu=mu)
    SDC.calc_sigmadc()
