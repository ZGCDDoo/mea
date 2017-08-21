from ..model import periodize_nambu, nambu
import numpy as np
import os, glob
from scipy.integrate import dblquad
from ..tools import fmanip
import json

def stiffness(fname, param_name="U"):

    (zn_vec, sEvec_c) = nambu.read_nambu_c(fname)

    with open("statsparams0.json") as fin:
        params = json.load(fin)

    mu = params["mu"][0]
    beta = params["beta"][0]
    tp = params["tp"][0]
    U = params[param_name][0]
    stiffness = 0.0
    stiffness_cum = 0.0
    stiffness_trace = 0.0

    model_sc = periodize_nambu.ModelNambu(1.0, tp, mu, 1.0j*zn_vec, sEvec_c)
    Y1 = model_sc.Y1Limit
    Y2 = model_sc.Y2Limit

    N_c = 4.0
    for ii in range(zn_vec.shape[0]):
        stiffness += 2.0/beta*1.0/(2.0*np.pi)**2*dblquad(model_sc.stiffness, -np.pi, np.pi, Y1, Y2, args=(ii,) )[0]
        stiffness_cum += 2.0/beta*1.0/(2.0*np.pi)**2*dblquad(model_sc.stiffness_cum, -np.pi, np.pi, Y1, Y2, args=(ii,) )[0]
        stiffness_trace += 2.0/beta*N_c/(2.0*np.pi)**2*dblquad(model_sc.stiffness_trace, -np.pi/2.0, np.pi/2.0, 
                                                                lambda x: -np.pi/2.0, lambda x: np.pi/2.0, args=(ii,) )[0]
        #print("stiffness = ", stiffness)

    #print("\n\n\n lattice stiffness \n\n.")
    #print("stiffness = ", stiffness)
    #print("\nstiffness_cum = ", stiffness_cum)
    fmanip.backup_file("stiffness.dat")
    np.savetxt("stiffness.dat", np.array([[stiffness, stiffness_cum, stiffness_trace]]))
    return (U, stiffness, stiffness_cum, stiffness_trace)


def stiff_walk(fname="self_moy.dat", param_name="U"):
    """walk a directory and get the stiffness for all subdirectories """
    folderlist = list(map(os.path.abspath, [dd for dd in os.listdir() if os.path.isdir(dd)] ))
    cwd = os.getcwd()

    stifflist = []
    for folder in folderlist:
        os.chdir(folder)
        os.chdir(glob.glob("Stats*")[0])
        result = stiffness(fname, param_name)
        stifflist.append(result)
        os.chdir(cwd)
        with open("output_stiff_walk.dat", mode="a") as fout:
            for element in result:
                fout.write(str(element)); fout.write(" ")
            fout.write("\n")
    
    #print("\nstifflist = ", stifflist)


