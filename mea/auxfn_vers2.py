#!/usr/bin/env python
# -*- coding: utf-8 -*-
#creating new branch MaxEntAux
"""
Same as auxfn.py, but passage to the irreducible basis is done after performing the auxiliary_fn.abs
Whereas in auxfn.py we transform the sE to the ir basis before doing the gfaux.
"""

import numpy as np
from scipy import linalg
from copy import deepcopy
import os
from .tools import fmanip
from .tools import kramerskronig as kk
from .model import green
from . import acon
from . import auxfn


class GFAuxC(auxfn.GFAux):
    """ A class that implements an Auxiliary green function. Input is given 
        as a sE file of a cluster, then it builds an auxiliary GF (it can do so
        for real or complex frequencies. However, imaginary frequencies is implicit).
        Afterwards, it extracts the sE_c on the real frequency axis."""

    fout_log = "log_eh.log"


    def __init__(self, zn_col=0, mu=0.0, fin_sE_to="self_moy.dat", 
                 fout_sE_ctow="self_ctow.dat", fout_gf_aux_to="gf_aux_to.dat", 
                 fout_log=fout_log, rm_sE_ifty=False, loc=None, non_loc=None,
                 non_loc_sign=None, non_loc_to_conjugate=None, delta=0.0):

        
        super(GFAuxC, self).__init__(zn_col=zn_col, mu=mu, fin_sE_to=fin_sE_to, fout_sE_ctow=fout_sE_ctow,
                                    fout_gf_aux_to=fout_gf_aux_to, fout_log=fout_log, rm_sE_ifty=rm_sE_ifty, loc=loc,
                                    non_loc=non_loc, non_loc_sign=non_loc_sign, non_loc_to_conjugate=non_loc_to_conjugate,
                                    delta=delta)
        self.sEvec_cw_list = []


    
    def build_gfvec_aux(self):
        """form an auxiliary green function matrix the same shape as the sE_c """
        
        sEvec_ir = self.sEvec_ir.copy()
        self.gfvec_aux_ir = np.zeros(sEvec_ir.shape, dtype=complex)
        self.gfvec_aux_c = self.gfvec_aux_ir.copy()
        assert(self.gfvec_aux_ir.shape == (self.zn_vec.shape[0] , 4 ,4))

        # iterate over matsubara freq.
        for (i, sE) in enumerate(self.sEvec_c):
            zz = (1.0j*self.zn_vec[i] + self.mu)
            self.gfvec_aux_c[i] = linalg.inv(zz*self.II - sE + self.sE_infty) if self.rm_sE_ifty \
                             else linalg.inv(zz*self.II - sE)

        self.gfvec_aux_ir = green.c_to_ir(self.gfvec_aux_c)

        # save the gfvec_aux_ir in a txt file   
         
        fmanip.backup_file(self.fout_gf_aux_to)    
        green.save_gf_ir(self.fout_gf_aux_to, self.zn_vec, self.gfvec_aux_ir)         
            
        
    

        
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
        gfvec_cw_list = map(green.ir_to_c, gfvec_irw_list)
        if fout_sE_ctow is None: fout_sE_ctow = self.fout_sE_ctow

        for (i, (w_vec, gf_cw)) in enumerate(zip(w_vec_list, gfvec_cw_list)):

            sEvec_cw = np.zeros(gf_cw.shape, dtype=complex)

            # iterate over matsubara freq.
            
            for (j, gf) in enumerate(gf_cw):
                ww = (w_vec[j] + self.mu)
                sEvec_cw[j] = (ww*self.II + self.sE_infty_w - linalg.inv(gf)) if self.rm_sE_ifty \
                                 else (ww*self.II - linalg.inv(gf))
            
            self.sEvec_cw_list.append(sEvec_cw)
            fout = fout_sE_ctow.split(".")[0] + str(i) + "." + fout_sE_ctow.split(".")[1] 

            fmanip.backup_file(fout)
            green.save_gf_c(fout, w_vec, sEvec_cw)


