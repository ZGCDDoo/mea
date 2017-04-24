#!/usr/bin/env python
# -*- coding: utf-8 -*-
#creating new branch MaxEntAux
"""

"""

import numpy as np
from scipy import linalg

import time
import subprocess
import re

import logging
from copy import deepcopy


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)

from mea import auxfn
from mea.tools import kramerskronig as kk
from mea import acon 
from mea.model import green, nambu


# Notations:
# gf_sb = green function superconducting block
# gf_normal = green function normal part

# algorithm:

# 0.) Read self energy of the cluster (nambu)
# 1.) Form normal sE_normal and superconducting block sEvec_sb
# 2.) call aux_fn to do analytic continuation of sE_normal
# 3.) Build gfvec_aux_sb and do acon on it 
# 4.) Rebuild the full sE_nambu_ir


class GFAuxSC():
    """ A class that implements an Auxiliary green function. Input is given 
        as a sE file of a cluster, then it builds an auxiliary GF (it can do so
        for real or complex frequencies. However, imaginary frequencies is implicit).
        Afterwards, it extracts the sEvec_c on the real frequency axis."""

    fout_log = "log_eh.log"

    def __init__(self, zn_col=0, fin_sE_to="self_moy_nambu.dat", 
                 fout_sE_ctow="self_nambu_ctow.dat", fout_gf_aux_to="gf_aux_sb.dat", 
                 fout_log=fout_log, rm_sE_ifty=False, loc_sb=None, non_loc_sb=None, non_loc_sign_sb=None, 
                 non_loc_to_conjugate_sb=None):
        """Initialize the GFAux object. The notation '_c' is used for a matrix cluster 
        representation, '_t' for a tabular form (in complex number and without the 
        matsubara grid), '_to' is used for a tabular form (with the matsubara grid
        and Re and Im seperated)

        Args:

        Keywords Args:
            zn_col (None||int): the col  which holds the matsubara frequencies
            fin_sE_to (str): the file which holds the sE_to file from Patrick's 
                Program 
            fout_sE_ctow (str): the file to which the sE_w_to (on the real axis) 
                will be written to
            fout_gf_aux_to (str): the file to which gf_aux_w_to (on the real axis) 
                will be written to
            fout_log (str): the file containing the logging info
            rm_sE_ifty (bool) : bool saying if the infty part of the sE should be removed
            loc (list(int)) : list containing the indices that are local in gf_aux_t
            non_loc (list(int)) : list containgin the indices that are non_local in gf_aux_t    
            
        Returns: None

        Raises:

         """ 

        # initilisation
        self.zn_col = zn_col
        self.fin_sE_to = fin_sE_to
        self.fout_sE_ctow = fout_sE_ctow
        self.fout_gf_aux_to = fout_gf_aux_to
        self.fout_log = fout_log
        self.rm_sE_ifty = rm_sE_ifty

        # here is gf_t for the superconducting block:
        # gf_sb[0, 0] = GB1UpB1Up ; gf_sb[1, 1] = -conjugate(GB2DownB2Down); gf_sb[0, 1].real = anoraml.real ; 
        # gf_sb[0 , 1].imag = anormal.imag ; gf_sb[1, 0].real = dagger(anormal).real ; gf_sb[0, 1].imag = dagger(anormal).imag
        self.loc_sb = [0, 1] if loc_sb is None else loc_sb    
        self.non_loc_sb = [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]] \
                          if non_loc_sb is None else non_loc_sb

        self.non_loc_sign_sb = [[1, 1, 1], [1, 1, 1], [-1, -1, 1], [-1, -1, 1]] \
                               if non_loc_sign_sb is None else non_loc_sign_sb

        self.non_loc_to_conjugate_sb = [[False, False, False], [False, False, False], 
                                        [True, True, False], [True, True, False]] \
                                        if non_loc_to_conjugate_sb is None else non_loc_to_conjugate_sb
        
        # check the sanity of the constructing parameters
        self.check_sanity()

        # attributes build afterwards by method calls
        # in matsubara freq. (a vector of matrixs, one matrix per matsubara freq.)
        (self.zn_vec, self.sEvec_c) = nambu.read_nambu_c(fin_sE_to, zn_col=0)
        self.sEvec_ir = nambu.c_to_ir(self.sEvec_c) 
        #self.sE_infty = nambu.read_nambu_infty(self.sEvec_ir)
        self.sEvec_normal_ir = nambu.get_normalup_ir(self.sEvec_ir)

        self.sEvec_sb = np.zeros((self.zn_vec.shape[0], 2, 2), dtype=complex)
        self.sEvec_sb[:, 0, 0] = self.sEvec_ir[:, 2, 2].copy(); self.sEvec_sb[:, 1, 1] = -np.conjugate(self.sEvec_ir[:, 3, 3].copy()) 
        self.sEvec_sb[:, 0, 1] = self.sEvec_ir[:, -1, 2].copy() 
        
        for (i, sE) in enumerate(self.sEvec_ir[:, -1, 2].copy()):
            self.sEvec_sb[i, 1, 0] = np.transpose(np.conjugate(sE))


        
        # attributes build afterwards by method calls.
        # a vector of matrixs (one matrix per matsubara frequency)
        self.gfvec_aux_sb = None  # build in method build_gfvec_aux

        self.acon = None  # build in method ac, is a class of acon
        self.aux_normal = None

        # on the real axis
        self.w_vec_list = []
        self.sEvec_irw_list = []
        #self.sE_infty_w = self.sE_infty  # the high frequency expansion zero term is the same for all complex zn_vec 
        self.Aw_t_list = []
        self.gf_aux_tw_list= []
        self.gf_aux_irw_list = []


        

        
    def check_sanity(self):
        """ """
        assert(self.rm_sE_ifty is True or self.rm_sE_ifty is False)
        assert(os.path.isfile(self.fin_sE_to)), "Ayaya, fin_sE_to does not exist"
        assert(type(self.zn_col) is int and self.zn_col >= 0), """Ayaya, 
        zn_col is not a valid int"""
        # Put other checks here

    
    def build_gfvec_aux(self):
        """form an auxiliary nambu function matrix the same shape as the sEvec_c """
        
        sEvec_sb = self.sEvec_sb.copy()
        self.gfvec_aux_sb = np.zeros(sEvec_sb.shape, dtype=complex)
        assert(self.gfvec_aux_sb.shape == (self.zn_vec.shape[0] , 2 ,2))

        # iterate over matsubara freq.
        II2 = np.eye(2, dtype=complex)
        for (i, sE) in enumerate(sEvec_sb):
            self.gfvec_aux_sb[i] = linalg.inv(1.0j*self.zn_vec[i]*II2 - sE) 

        # save the gfvec_aux_sb
        gfvec_aux_sb_t = np.zeros((self.zn_vec.shape[0], 10))

        gfvec_aux_sb_t[:, 0] = self.zn_vec 
        gfvec_aux_sb_t[:, 1] = self.gfvec_aux_sb[:, 0, 0].real; gfvec_aux_sb_t[:, 2] = self.gfvec_aux_sb[:, 0, 0].imag
        gfvec_aux_sb_t[:, 3] = self.gfvec_aux_sb[:, 1, 1].real; gfvec_aux_sb_t[:, 4] = self.gfvec_aux_sb[:, 1, 1].imag
        gfvec_aux_sb_t[:, 5] = self.gfvec_aux_sb[:, 0, 1].real; gfvec_aux_sb_t[:, 6] = self.gfvec_aux_sb[:, 0, 1].imag
        gfvec_aux_sb_t[:, 7] = self.gfvec_aux_sb[:, 1, 0].real; gfvec_aux_sb_t[:, 8] = self.gfvec_aux_sb[:, 1, 0].imag

        np.savetxt(self.fout_gf_aux_to, gfvec_aux_sb_t)  
        
    def ac(self, fin_OME_default="OME_default.dat", fin_OME_other="OME_other.dat", 
           fin_OME_input="OME_input.dat"):
        """Perform the analytic continuation (acon) to find gf_aux_w on the real 
        frequency axis """

        # acon of the normal part
        green.save_gf_c("self_moy_normal.dat", self.zn_vec, green.ir_to_c(self.sEvec_normal_ir))        
        self.aux_normal = auxfn.GFAux(fin_sE_to="self_moy_normal.dat")
        self.aux_normal.ac(fin_OME_default, fin_OME_other, fin_OME_input) ; self.aux_normal.get_sEvec_w_list()
        self.sE_w_ir_normal_list = self.aux_normal.sEvec_irw_list
        self.w_vec_list = self.aux_normal.w_vec_list



        if self.gfvec_aux_sb is None: self.build_gfvec_aux()
        
        # represent the gfvec_aux_sb as a tabular form of complex numbers without the matsubara grid
        gf_t = np.zeros((self.zn_vec.shape[0], 6), dtype=complex)
        gf_t[:, 0] = self.gfvec_aux_sb[:, 0 ,0].copy() ; gf_t[:, 1] = self.gfvec_aux_sb[:, 1 , 1].copy(); 
        gf_t[:, 2]  = self.gfvec_aux_sb[:, 0, 1].real.copy() ; gf_t[:, 3] = 1.0j*self.gfvec_aux_sb[:, 0, 1].imag.copy()
        gf_t[:, 4] = self.gfvec_aux_sb[:, 0 , 1].real.copy(); gf_t[:, 5] = 1.0j*self.gfvec_aux_sb[:, 1, 0].imag.copy()

        #Write the w_vec grid in the OME_input files
        for (i, w_vec) in enumerate(self.w_vec_list):
            # Write the file to disk, then the filename to OME_input + str(i) + ".dat"
            if w_vec is None:
                continue

            w_vec_file = "w_n" + str(i) + ".dat"
            np.savetxt(w_vec_file, w_vec)
            fin_file = fin_OME_input.split(".")[0] + str(i) + "." + fin_OME_input.split(".")[1]
            
            with open(fin_file) as fin:
                fin_str = fin.read()
                if "real frequency grid file" in fin_str:
                    fin_str = fin_str.replace("real frequency grid file:", "real frequency grid file:" + w_vec_file)
                else:
                    fin_str += "\nreal frequency grid file:" +  w_vec_file
            
            fin_file_sb = fin_OME_input.split(".")[0] + "_sb_" + str(i) + fin_OME_input.split(".")[1]        
            with open(fin_file_sb, mode="w") as fin:
                fin.write(fin_str)

        self.acon = acon.ACon(gf_t, self.zn_vec, 
                              fin_OME_default=fin_OME_default, fin_OME_other=fin_OME_other, 
                              fin_OME_input=fin_OME_input, loc=self.loc_sb, non_loc=self.non_loc_sb,
                              non_loc_sign=self.non_loc_sign_sb,
                              non_loc_to_conjugate=self.non_loc_to_conjugate_sb, NN=2.0)


        self.acon.acon()
        self.Aw_t_list = deepcopy(self.acon.Aw_t_list)

#        for (i, (w_vec, Aw_aux_t)) in enumerate(zip(self.acon.w_vec_list, self.acon.Aw_vec_t_list)):
#            if w_vec is None:
#                continue
#            gf_aux_w_t = np.zeros(Aw_aux_t.shape, dtype=complex)
#            for i in range(Aw_aux_t.shape[1]):
#                gf_aux_w_t[:, i] = -0.5*(kk.KramersKroning(w_vec, Aw_aux_t[:, i]) + 1.0j*Aw_aux_t[:, i])
#            self.gf_aux_tw_list.append(gf_aux_w_t)
#            self.w_vec_list.append(w_vec)    
#            self.gf_aux_irw_list.append(green.t_to_ir(w_vec, gf_aux_w_t)) 

 

        
    def get_sEvec_w_list(self, fout_sE_ctow=None):
        """ """
        pass
