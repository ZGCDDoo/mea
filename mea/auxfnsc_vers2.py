#!/usr/bin/env python
# -*- coding: utf-8 -*-
#creating new branch MaxEntAux
"""

"""

import numpy as np
from scipy import linalg
from copy import deepcopy
import os
from .tools import fmanip
from .tools import kramerskronig as kk
from .model import nambu, periodize_nambu
from . import acon



class GFAuxSC():
    """ A class that implements an Auxiliary green function. Input is given 
        as a sE file of a cluster, then it builds an auxiliary GF (it can do so
        for real or complex frequencies. However, imaginary frequencies is implicit).
        Afterwards, it extracts the sE_c on the real frequency axis."""

    fout_log = "log_eh.log"

    def __init__(self, zn_col=0, mu=0.0, fin_sE_to="self_moy.dat", 
                 fout_sE_ctow="self_ctow.dat", fout_gf_aux_to="gf_aux_to_sc.dat", 
                 fout_log=fout_log, loc=None, non_loc=None,
                 non_loc_sign=None, non_loc_to_conjugate=None, delta=0.0):
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
        self.mu = mu
        self.fin_sE_to = fin_sE_to
        self.fout_sE_ctow = fout_sE_ctow
        self.fout_gf_aux_to = fout_gf_aux_to
        self.fout_log = fout_log
        self.loc = (0, 1, 2, 3) if loc is None else loc    
        self.non_loc = ((0, 1, 2, 3, 4), (2, 3, 5)) if non_loc is None else non_loc
        self.non_loc_sign = ((1, 1, 1, 1, 1), (1, -1, 1)) if non_loc_sign is None else non_loc_sign
        self.non_loc_to_conjugate = ((False, False, False, False, False), (False, True, False)) if non_loc_to_conjugate is None else non_loc_to_conjugate
        self.delta = delta
        # check the sanity of the constructing parameters
        self.check_sanity()

        # attributes build afterwards by method calls
        # in matsubara freq. (a vector of matrixs, one matrix per matsubara freq.)
        (self.zn_vec, self.sEvec_c) = nambu.read_nambu_c(fin_sE_to, zn_col=0)
        self.sEvec_ir = nambu.c_to_ir(self.sEvec_c) 

        
        # attributes build afterwards by method calls.
        # a vector of matrixs (one matrix per matsubara frequency)
        self.gfvec_aux_ir = None  # build in method build_gfvec_aux
        self.gfvec_aux_c = None   # build in method build_gfvec_aux

        self.acon = None  # build in method ac, is a class of acon
        
        # on the real axis
        self.w_vec_list = []
        self.sEvec_irw_list = []
        self.Aw_t_list = []
        self.gfvec_aux_irtw_list= []
        self.gfvec_aux_irw_list = []


        

        
    def check_sanity(self):
        """ """
        assert(os.path.isfile(self.fin_sE_to)), "Ayaya, fin_sE_to does not exist"
        assert(type(self.zn_col) is int and self.zn_col >= 0), """Ayaya, 
        zn_col is not a valid int"""
        # Put other checks here

    
    def build_gfvec_aux(self):
        """form an auxiliary green function matrix the same shape as the sE_c """
        
        modeln = periodize_nambu.ModelNambu(0.0, 0.0, self.mu, 1.0j*self.zn_vec, self.sEvec_ir)
        self.gfvec_aux_ir = np.zeros(self.sEvec_ir.shape, dtype=complex)
        assert(self.gfvec_aux_ir.shape == (self.zn_vec.shape[0] , 8 ,8))

        # iterate over matsubara freq.
        for ii in range(self.sEvec_ir.shape[0]):
            self.gfvec_aux_ir[ii] = modeln.build_gf_ktilde(0.0, 0.0, ii)

        self.gfvec_aux_c = nambu.ir_to_c(self.gfvec_aux_ir)

        # save the gfvec_aux_ir in a txt file   
         
        fmanip.backup_file(self.fout_gf_aux_to)    
        nambu.save_nambu_ir(self.fout_gf_aux_to, self.zn_vec, self.gfvec_aux_ir)         
            
        
    def ac(self, fin_OME_default="OME_default.dat", fin_OME_other="OME_other.dat", 
           fin_OME_input="OME_input.dat"):
        """Perform the analytic continuation (acon) to find gf_aux_w on the real 
        frequency axis """
        if self.gfvec_aux_ir is None: self.build_gfvec_aux()
        
        # represent the gfvec_aux_ir as a tabular form of complex numbers without the matsubara grid
        gf_aux_irt = nambu.ir_to_t(self.zn_vec, self.gfvec_aux_ir)

                  
        self.acon = acon.ACon(gf_aux_irt, self.zn_vec, 
                    fin_OME_default=fin_OME_default, fin_OME_other=fin_OME_other, 
                    fin_OME_input=fin_OME_input, loc=self.loc, non_loc=self.non_loc,
                    non_loc_sign=self.non_loc_sign,
                    non_loc_to_conjugate=self.non_loc_to_conjugate, delta=self.delta)
        
        # print("IN gf_aux.acon() \n")
        self.acon.acon()
        # print("after gf_aux.acon() \n")
        self.Aw_t_list = deepcopy(self.acon.Aw_t_list)
        
        # force anti-symmetry on the anomalous spectral function:
        for (ii, Aw_t) in enumerate(self.Aw_t_list):
            Aw_t[:, -1] = 0.5*(Aw_t[:, -1] - Aw_t[:, -1][::-1])


        for (i, (w_vec, Aw_aux_t)) in enumerate(zip(self.acon.w_vec_list, self.acon.Aw_t_list)):
            if w_vec is None:
                continue
            gfvec_aux_irtw = np.zeros(Aw_aux_t.shape, dtype=complex)
            for i in range(Aw_aux_t.shape[1]):
                gfvec_aux_irtw[:, i] = -0.5*(kk.KramersKroning(w_vec, Aw_aux_t[:, i]) + 1.0j*Aw_aux_t[:, i])
            self.gfvec_aux_irtw_list.append(gfvec_aux_irtw)
            self.w_vec_list.append(w_vec)    
            self.gfvec_aux_irw_list.append(nambu.t_to_irw(w_vec, gfvec_aux_irtw))
 

        
    def get_sEvec_w_list(self, fout_sE_ctow=None):
        """Extract  sE_w on the real axis from the real axis gf_aux_w

        Args:

        Keywords Args:
            zn_col (None||int): the col  which holds the matsubara frequencies
            fin_sE_c (str): the file which holds the sE_c file from Patrick's 
                Program 
            fout_sE_c (str): the file to which the sE_c_w (on the real axis) 
                will be written to
            fout_aux_fn (str): the file to which gf_aux_w (on the real axis) 
                will be written to
            
        Returns: None

        Raises:

         """ 

        gfvec_irw_list = deepcopy(self.gfvec_aux_irw_list) ; w_vec_list = deepcopy(self.w_vec_list)
        
        if fout_sE_ctow is None: fout_sE_ctow = self.fout_sE_ctow

        for (i, (w_vec, gf_irw)) in enumerate(zip(w_vec_list, gfvec_irw_list)):

            sEvec_irw = np.zeros(gf_irw.shape, dtype=complex)

            # iterate over matsubara freq.
            
            for (j, gf) in enumerate(gf_irw):
                sEvec_irw[j] -= linalg.inv(gf)
                sEvec_irw[j, :4:, :4:] += (w_vec[j] + self.mu)
                sEvec_irw[j, 4::, 4::] -= (-w_vec[j] + self.mu)
            
            self.sEvec_irw_list.append(sEvec_irw)
            fout = fout_sE_ctow.split(".")[0] + str(i) + "." + fout_sE_ctow.split(".")[1] 

            fmanip.backup_file(fout)
            nambu.save_nambu_c(fout, w_vec, nambu.ir_to_c(sEvec_irw))


