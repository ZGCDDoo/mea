#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg

import time
import subprocess
import re

import logging
from copy import deepcopy


import os, sys, inspect


from . import auxfn
from .tools import kramerskronig as kk
from . import acon 
from .model import green, nambu, periodize_nambu



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

        
        # check the sanity of the constructing parameters
        self.check_sanity()

        # attributes build afterwards by method calls
        # in matsubara freq. (a vector of matrixs, one matrix per matsubara freq.)
        (self.zn_vec, self.sEvec_c) = nambu.read_nambu_c(fin_sE_to, zn_col=0)
        self.sEvec_ir = nambu.c_to_ir(self.sEvec_c) 

        # attributes build afterwards by method calls.
        # a vector of matrixs (one matrix per matsubara frequency)

        self.acon = None  # build in method ac, is a class of acon

        # on the real axis
        self.w_vec_list = []
        self.sEvec_irw_list = []
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
        
        # iterate over matsubara freq.
        self.modeln = periodize_nambu.ModelNambu(0.0, 0.0, self.mu, 1.0j*self.zn_vec, self.sEvec_ir)
        self.gfvec_aux_ir = np.zeros(self.sEvec_ir.shape, dtype=complex)

        for ii in range(self.sEvec_ir.shape[0]):
            self.gfvec_aux_ir[ii] = self.modeln.periodize_nambu(0.0, 0.0, ii) 


