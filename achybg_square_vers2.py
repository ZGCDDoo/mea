# coding: utf-8

import numpy as  np
from mea import acon_square as acon
import shutil
from mea import kramerskronig as kk
from copy import deepcopy
import json
import sys
import os


def extract_parameter_bqsubmit(param_file, str_parameter) :
    # str_parameter = 'U' for instance
    # param_file = 'param.in' for instance
    param_file_lines = open(param_file, 'r')
    for line in param_file_lines :
        if line.strip() == '' :
            continue
        elif line.split()[0] == str_parameter :
            parameter = float(line.split()[2])
    return parameter


def average_files(DATA_DIR, iter_start, iter_end):
        
    fname_hyb = os.path.join(DATA_DIR, "Hyb" + str(iter_start) + ".json")
    with open(fname_hyb) as fin:
        params_fin = json.load(fin)
        hybvec = np.array(params_fin["0Pi"]["real"], dtype=complex) + 1.0j*np.array(params_fin["0Pi"]["imag"], dtype=complex )


    fname_self = os.path.join(DATA_DIR, "Self" + str(iter_start) + ".json")
    with open(fname_self) as fin:
        params_fin = json.load(fin)
        selfvec = np.array(params_fin["0Pi"]["real"], dtype=complex) + 1.0j*np.array(params_fin["0Pi"]["imag"], dtype=complex)

    for ii in range(iter_start + 1, iter_end + 1):

        fname_hyb = os.path.join(DATA_DIR, "Hyb" + str(ii) + ".json")
        with open(fname_hyb) as fin:
            params_fin = json.load(fin)
            hybvec += np.array(params_fin["0Pi"]["real"]) + 1.0j*np.array(params_fin["0Pi"]["imag"] )


        fname_self = os.path.join(DATA_DIR, "Self" + str(ii) + ".json")
        with open(fname_self) as fin:
            params_fin = json.load(fin)
            selfvec += np.array(params_fin["0Pi"]["real"], dtype=complex) + 1.0j*np.array(params_fin["0Pi"]["imag"], dtype=complex)

    NN = iter_end - iter_start + 1
    selfvec /= NN
    hybvec /= NN
    return (selfvec, hybvec)




def extract_realaxis(achyb, acgf):

    hybvec_w_list = []; gfvec_w_list = []
    w_vec_list = []

    for (ii, (w_vec, Aw_t)) in enumerate(zip(achyb.w_vec_list, achyb.Aw_t_list)):
        if w_vec is None:
            hybvec_w_list.append(None)
            w_vec_list.append(None)
        else:
            hybvec_w = np.zeros(Aw_t.shape, dtype=complex)
            hybvec_w = -0.5*(kk.KramersKroning(w_vec, Aw_t[:,0]) + 1.0j*Aw_t[:,0])
            hybvec_w_list.append(hybvec_w)
            w_vec_list.append(w_vec)


    for (ii, (w_vec, Aw_t)) in enumerate(zip(acgf.w_vec_list, acgf.Aw_t_list)):
        if w_vec is None:
            gfvec_w_list.append(None)
        else:
            gfvec_w = np.zeros(Aw_t.shape, dtype=complex)
            gfvec_w = -0.5*(kk.KramersKroning(w_vec, Aw_t[:,0]) + 1.0j*Aw_t[:,0])
            gfvec_w_list.append(gfvec_w)

    return (w_vec_list, hybvec_w_list, gfvec_w_list)



def extract_selfw(w_vec_list, hybvec_w_list, gfvec_w_list, mu):

    for (n, (gfvec_w, w_vec)) in enumerate(zip(gfvec_w_list, w_vec_list)):
        if gfvec_w is None or hybvec_w_list[n] is None:
            continue
        else:
            selfvec_w = -1.0/gfvec_w + w_vec + mu - hybvec_w_list[n]
            fname = "self_w" + str(n) + ".dat"
            np.savetxt(fname, np.transpose([w_vec, selfvec_w.real, selfvec_w.imag]))
    
    return None


def main(iter_start, iter_end, N):

    beta = extract_parameter_bqsubmit("bqsubmit.dat", 'beta')
    t = 1.0
    tp = extract_parameter_bqsubmit("bqsubmit.dat", 'tp')
    U = int(extract_parameter_bqsubmit("bqsubmit.dat", 'U'))
    #tpp = params_fin["tpp"][0]
    if abs(tp) < 0.02:
        tp = int(tp)

    DATA_DIR = "VH_tp" + str(tp) + "_n" + str(N) + ".BQ"


    [interation, list_mu] = np.genfromtxt(DATA_DIR + "/mu.dat").T
    mu = np.mean(list_mu[iter_start:iter_end])
    (selfvec, hybvec) = average_files(DATA_DIR, iter_start, iter_end)
    znvec = np.array([(2.0*n + 1.0)*np.pi/beta for n in range(selfvec.shape[0])])

    achyb = acon.ACon(hybvec[:, np.newaxis], znvec) ; achyb.run_acon()
    shutil.move("Result_OME", "Result_OME_hyb")

    gfaux = 1.0/(1.0j*znvec + mu - hybvec - selfvec)
    acgf = acon.ACon(gfaux[:, np.newaxis], znvec); acgf.run_acon()
    shutil.move("Result_OME", "Result_OME_gfaux")
    (w_vec_list, hybvec_w_list, gfvec_w_list) = extract_realaxis(achyb, acgf)
    extract_selfw(w_vec_list, hybvec_w_list, gfvec_w_list, mu)


def correct_selfw(mu, fnamehyb, fnamegf):
    
    Awhyb = np.loadtxt(fnamehyb)
    w_vec = Awhyb[:, 0]
    hybvec_w = -0.5*(kk.KramersKroning(w_vec, Awhyb[:,1]) + 1.0j*Awhyb[:,1])

    Awgf = np.loadtxt(fnamegf)
    gfvec_w = -0.5*(kk.KramersKroning(w_vec, Awgf[:,1]) + 1.0j*Awgf[:,1])

    selfvec_w = -1.0/gfvec_w + w_vec + mu - hybvec_w
    np.savetxt("self_w_corrected.dat", np.transpose([w_vec, selfvec_w.real, selfvec_w.imag]))
    return None

def correct_all(mu):

    for ii in range(10):
        fnamehyb = os.path.join("Result_OME_hyb", os.path.join(str(ii),  "Aw_t.dat"))
        fnamegf = os.path.join("Result_OME_gfaux", os.path.join(str(ii), "Aw_t.dat"))
        if not os.path.isfile(fnamegf):
            break
        correct_selfw(mu, fnamehyb, fnamegf)
        shutil.move("self_w_corrected.dat", "self_w_corrected" + str(ii) + ".dat")


if __name__ == "__main__":

    iter_start = int(sys.argv[1])
    iter_end = int(sys.argv[2])
    N = sys.argv[3]
    main(iter_start, iter_end, N)