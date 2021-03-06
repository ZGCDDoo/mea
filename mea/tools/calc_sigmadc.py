from mea.model.triangle import Triangle as Model
from mea.model.io_triangle import IOTriangle as green
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
    
    (wvec, sEvec_c) = green().read_green_c(fname)
    model = Model(1.0, 0.4 , 0.0, mu, wvec, sEvec_c)
    sdc = sigmadc.SigmaDC(model, beta=beta)
    sdc.calc_sigmadc()
