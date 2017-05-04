#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Branch MaxEntAux

import numpy as np
import os

import subprocess
import re
import logging
import warnings
import shutil
from copy import deepcopy

from .tools import fmanip




class ACon():
    """ A class that implements the analytic continuation (acon) of a green
        function (gf) the green function is written as a tabular form (gf_t) of all independant
        parts of the green function of the cluster of the irreducible representation (gf_ir).
        Calculates and saves the real axis frequency grid and the gf_t(w) for each run
        with different inputt files for OmegaMaxEnt ('OME_input.dat')."""

    # Defaults for AnisotropicTriangularC2v (not SC)
    loc_default = [0, 1, 2, 3]
    non_loc_default = [[0, 1, 2, 3, 4]]
    non_loc_sign_default = [[1, 1, 1, 1, 1]]
    non_loc_to_conjugate_default = [[False, False, False, False, False]]
    fout_log = "log_mea2_0.dat"

    def __init__(self, gf_t, zn_vec, fout_Aw_t="Aw_t.dat",
                 fin_OME_default="OME_default.dat", fin_OME_other="OME_other.dat",
                 fin_OME_input="OME_input.dat", loc=loc_default,
                 non_loc=non_loc_default, non_loc_sign=non_loc_sign_default, non_loc_to_conjugate=non_loc_to_conjugate_default, fout_log=fout_log,
                 NN = None):
        """Initialize the acon object.

        Args:

        Keywords Args:
            gf_t (np.array(dtype=complex)): an auxiliary (Reza Style) green function in the
                matrix cluster sites in a tabular form in complex form
            zn_vec (np.array(dype=complex)): the matsubara frequencies
            fout_A_w (str): the file to which to write the auxilary spectral
                function in the matrix cluster sites

        Returns:

        Raises:
        """

        # initialisation
        self.gf_t = gf_t.copy()
        self.zn_vec = zn_vec.copy()
        self.fout_Aw_t = fout_Aw_t

        self.loc = deepcopy(loc)  # the local or diagonal green functions in gf_c
        self.non_loc = deepcopy(non_loc)  # the non-loc or non-diagonal green functions in gf_c
        self.non_loc_sign = deepcopy(non_loc_sign)
        self.non_loc_to_conjugate = deepcopy(non_loc_to_conjugate)
        self.NN = NN

        self.outdir = "Result_OME" # output directory for the results
        self.tmp_file = "tmp_green.dat"
        self.tmp_file_error = "tmp_green_error.dat"

        fmanip.backup_file(fout_log)
        logging.basicConfig(filename=fout_log, level=logging.DEBUG)

        # Read the default files for OmegaMaxEnt = OME
        assert(os.path.isfile(fin_OME_default)), "Ayaya, OME_default.dat does not exist"
        assert(os.path.isfile(fin_OME_other)), "Ayaya, OME_other.dat does not exist"

        with open(fin_OME_default) as fd, open(fin_OME_other) as fo:
            self.OME_default = fd.readlines()
            self.OME_other = fo.readlines()

        self.OME_input = []
        self.OME_input = fmanip.build_file_list(fin_OME_input)
        #self.build_OME_input_list(fin_OME_input)

        # attributes build afterwards by method calls
        # (the desired output in fact of acon.py)
        self.w_vec_list = []
        self.w_vec_file = "w_n.dat"
        self.gf_aux_t = None
        self.Aw_aux_t_list = []
        self.Aw_t_list = []

        self.check_sanity()

    # TO DO: Put this method in a file_maipulator module (fmanip)
    def build_OME_input_list(self, fin_OME_input):
        """ """
        abs_dir = os.path.split(fin_OME_input)[0] ; fname = os.path.split(fin_OME_input)[1].split(".")[0]
        ext = os.path.split(fin_OME_input)[1].split(".")[1]
        iteration = 0
        fname_tmp = os.path.join(abs_dir, fname + str(iteration) + "." + ext)
        while os.path.isfile(fname_tmp):
            with open(fname_tmp) as indata:
                self.OME_input.append(indata.readlines())
            iteration += 1
            fname_tmp = os.path.join(abs_dir, fname + str(iteration) + "." + ext)

        if self.OME_input == []:
            print("Invalid name format for OME_input: ") ; raise ValueError
        
        


    def check_sanity(self):
        """ """

        try:
            assert(len(self.loc) + len(self.non_loc) == self.gf_t.shape[1]), "Ayaya, problem with shape of the loc and non_loc"
        except AssertionError:
            msg = "len(loc) + len(non_loc) != gf_t.shape[1]. Is this volontary?"
            warnings.warn(msg)
            logging.warning(msg)

        assert(self.zn_vec.shape[0] == self.gf_t.shape[0]), "Ayaya, problem with shape of zn_vec and of the greens."
        assert(self.gf_t.dtype == np.dtype("complex128"))
        assert(len(self.non_loc) == len(self.non_loc_to_conjugate) ), "Ayaya, problem, len of non_loc and non_loc_to_conjugate are not the same."
        assert(len(self.non_loc_sign) == len(self.non_loc_to_conjugate) ), "Ayaya, problem, len of non_loc_sign and non_loc_to_conjugate are not the same."
        
        # make sure that everything in OME_input is valid for OME (that it exists in OME_default.dat)
        for indata in self.OME_input:
            for line in indata:
                tmp = line.split(":")
                regx = ""
                for part in tmp[:-1]:
                    regx += part + ":"
                assert(regx + "\n" in self.OME_default), "Ayaya, OME_input.dat invalid at : " + line


    def build_gf_aux_t(self):
        """build the gf_aux_t to do acon"""

        gf_t = np.copy(self.gf_t)

        # 1.) Build the gf_aux_t
        gf_aux_t = np.zeros((gf_t.shape[0], len(self.loc) + len(self.non_loc)), dtype=complex)

        for (i, num) in enumerate(self.loc):
            gf_aux_t[:, i] = gf_t[:, num].copy()
        for (ii, nums) in enumerate(self.non_loc):
            to_conjugate_list = self.non_loc_to_conjugate[ii]
            sign_list = self.non_loc_sign[ii]
            l = ii + len(self.loc)
            N = len(nums) - 1 if self.NN is None else self.NN
            for (num, sign, toconjugate) in zip(nums[:-1:], sign_list[:-1:], to_conjugate_list[:-1:]):
                sgn = np.sign(sign)
                gf_tmp = np.conjugate(gf_t[:, num]) if toconjugate else gf_t[:, num].copy() 
                gf_aux_t[:, l] += 1/N * sgn * gf_tmp
            sgn = np.sign(sign_list[-1])
            gf_tmp = np.conjugate(gf_t[:, nums[-1]]) if to_conjugate_list[-1] else gf_t[:, nums[-1]].copy()
            gf_aux_t[:, l] += 2.0/N * sgn* gf_tmp

        #delta = 0.0#10**(-3.0)
        #cst_err = delta * gf_aux_t[0, 0].imag * np.random.choice([-1.0, 1.0], size=gf_aux_t.shape)
        self.gf_aux_t = gf_aux_t.copy() #* (1.0 + cst_err)
        #self.gf_aux_error_t = np.absolute(cst_err) * (1.0 + 1.0j)

        # save the gf_aux_t_acon.dat in a file
        tmp = np.zeros((self.gf_aux_t.shape[0], 2*self.gf_aux_t.shape[1] + 1))
        tmp[:, 0] = self.zn_vec.copy()
        tmp[:, 1::2] = self.gf_aux_t.real.copy()
        tmp[:, 2::2] = self.gf_aux_t.imag.copy()
        fout_gf_aux_t = "gf_aux_t_acon.dat"
        
        fmanip.backup_file(fout_gf_aux_t)
        np.savetxt(fout_gf_aux_t, tmp)


    def build_Aw_t(self, Aw_aux_t):
        """build the A_w_t of the original gf_t by doing the inverse transformations of 'build_gf_aux_t'
        with Aw_aux_t """

        Aw_aux_t = np.copy(Aw_aux_t)
        Aw_t = np.zeros(Aw_aux_t.shape, dtype="float64")

        for (i, num) in enumerate(self.loc):
            Aw_t[:, num] = Aw_aux_t[:, i].copy()
        for (ii, nums) in enumerate(self.non_loc):
            to_reverse_list = self.non_loc_to_conjugate[ii]
            sign_list = self.non_loc_sign[ii]
            l = ii + len(self.loc)
            assert(l == nums[-1])
            N = len(nums) - 1.0 if self.NN is None else self.NN
            sgn = np.sign(sign_list[-1]) # 1 if nums[-1] is 0 else np.sign(nums[-1])
            Aw_tmp = Aw_aux_t[:, l][::-1].copy() if to_reverse_list[-1] else Aw_aux_t[:, l].copy()
            Aw_t[:, nums[-1]] += N/2.0*sgn*Aw_tmp
            for (num, sign, toreverse) in zip(nums[:-1:], sign_list[:-1:], to_reverse_list[:-1:]):
                sgn2 = np.sign(sign)
                Aw_tmp = -Aw_aux_t[:, num][::-1].copy() if toreverse else Aw_aux_t[:, num].copy()
                Aw_t[:, nums[-1]] -= sgn*0.5*sgn2*Aw_tmp

        return Aw_t.copy()


    def acon(self):
        """Perform the analytic continuation (acon) to find gf_t on the real frequency axis """
        # Note the abbreviation: OME = OmegaMaxEnt
        self.build_gf_aux_t() ; gf_aux_t = np.copy(self.gf_aux_t)

        folder_names = ["OmegaMaxEnt_final_result", "OmegaMaxEnt_files"]

        fmanip.backup_folder(self.outdir)
        os.mkdir(self.outdir)

        for i in range(len(self.OME_input)):
            print("\niteration OME_input = ", i)
            logging.info("\niteration OME_input = %s ", i)

            for (j, gf) in enumerate(np.transpose(gf_aux_t)):
                logging.info("\niteration = %s ", j)
                print("\niteration = ", j)

                self.acon_preprocess()
                np.savetxt(self.tmp_file, np.array([self.zn_vec, gf.real, gf.imag]).T)
                #np.savetxt(self.tmp_file_error, np.array([self.zn_vec, self.gf_aux_error_t[:, j].real, self.gf_aux_error_t[:, j].imag]).T)
                assert os.path.isfile(self.tmp_file), "ayaya, tmp_green.dat does not exist"

                # 3.1) Modify the elements in the input_file and call OmegaMaxEnt
                self.make_OME_file(iter_OME_input=i, iteration=j); self.call_OmegaMaxEnt(to_log=True) #call it with a given OME_input_iter.dat

                # 3.3) Read the corresponding Aw that OmegaMaxEnt gave
                Aw_tmp = self.read_Aw_from_OME()
                if Aw_tmp is None:
                    Aw_aux_t = None
                    w_vec = None
                    break

                if j == 0:
                    w_vec = Aw_tmp[:, 0]
                    np.savetxt(self.w_vec_file, w_vec)
                    Aw_aux_t = np.zeros((w_vec.shape[0], self.gf_t.shape[1]))
                    
                    cur_write_dir = os.path.join(self.outdir, str(i)); os.mkdir(cur_write_dir)
                    
                Aw_aux_t[:, j] = Aw_tmp[:, 1]
                          
                for folder in folder_names:
                    if os.path.isdir(folder):
                        shutil.move(folder, os.path.join(cur_write_dir, folder + str(j)) )
    
                shutil.move(self.tmp_file, self.tmp_file.split(".")[0] + str(j) + "." + self.tmp_file.split(".")[1])                     
            
            Aw_t = None if Aw_aux_t is None else self.build_Aw_t(Aw_aux_t)
            self.Aw_aux_t_list.append(Aw_aux_t)
            self.w_vec_list.append(w_vec)
            self.Aw_t_list.append(Aw_t)
            
            if Aw_aux_t is not None:
                self.save_Aw_t(w_vec, Aw_t, os.path.join(cur_write_dir, self.fout_Aw_t))
            self.cleanup()


    def read_Aw_from_OME(self):
        """ """
        
        try:
            Aw = None
            file = [file for file in os.listdir("OmegaMaxEnt_final_result") \
                   if os.path.isfile(os.path.join("OmegaMaxEnt_final_result", file)) and \
                   "optimal_spectral_function_tem" in file]
            assert(len(file) == 1)
            Aw = np.loadtxt(os.path.join("OmegaMaxEnt_final_result", file[0]))
            assert(Aw.shape[1] == 2), "ayaya, wrong shape for A_w_tmp"
        except (AssertionError, FileNotFoundError) as err: 
            print("Error is {0}".format(err))
            warn_msg = "Ayayay, OmegaMaxEnt did not find the spectral function, Error,'omegaMaxEnt_final_result' does not exist."
            logging.warning(warn_msg)
            warnings.warn(warn_msg)

        return Aw

    def acon_preprocess(self):
        """ """
        OME_fin = "OmegaMaxEnt_input_params.dat"
        files_names = ["OmegaMaxEnt_input_params.dat", "newline.txt", self.tmp_file, self.tmp_file_error] #, "w_vec.dat"]
        folder_names = ["OmegaMaxEnt_final_result", "OmegaMaxEnt_files"]
        
        for file in files_names:
            if os.path.isfile(file):
                os.remove(file)
        
        with open("newline.txt", mode="w") as nl:
            nl.write("\n")

        for folder in folder_names:
            fmanip.backup_folder(folder)
                
        # 2.) Create the default file for OME inputt
        self.call_OmegaMaxEnt(indata=b"\n", timeout=10, to_log=False) #creates the file license_infodisplayed
        self.call_OmegaMaxEnt(indata=b"\n", timeout=10, to_log=False) #creates the file noticed_displayed


    # Try to put this in fmanip.py in near futur. Also clean up and remove unecessary parameters.
    def make_OME_file(self, iter_OME_input, iteration,  fin="OmegaMaxEnt_input_params.dat", fout="OmegaMaxEnt_input_params.dat"):
        """ """
        with open(fin, mode="r") as OME_input:
            # 3.1) Modify the elements in the input_file
            OME_input_s = OME_input.read()
            OME_input_s = re.sub(r"data file:", "data file:" + self.tmp_file, OME_input_s)
            OME_input_s = re.sub(r"error file:", "error file:" + self.tmp_file_error, OME_input_s)

            for line in self.OME_input[iter_OME_input]:
                line_strip = line.rstrip()
                tmp = line.split(":")
                regx = ""
                for part in tmp[:-1]:
                    regx += part + ":"
                regx = re.escape(regx)
                OME_input_s = re.sub(regx, line_strip, OME_input_s)

            if iteration >= 1 and "real frequency grid file" not in " ".join(self.OME_input[iter_OME_input]):
                OME_input_s = re.sub(r"real frequency grid file:", "real frequency grid file:" + self.w_vec_file, OME_input_s)
        #print(OME_input_s)

        with open(fout, mode="w") as OME_output:
            OME_output.write(OME_input_s)


    def call_OmegaMaxEnt(self, indata=None, timeout=None, to_log=False):
        """ """
        try:
            stdout = None
            stdout = subprocess.check_output(args=["OmegaMaxEnt"], shell=False,
                                            input=indata, timeout=timeout).decode()
        except (FileNotFoundError, OSError) as err: #timout exception, then filenotfounderror (if OmegaMaxEnt does not exist, then catch all othre)
            print("OS error: {0}".format(err)) ; raise
        except subprocess.TimeoutExpired as err:
            print("preprocess timeout error: {0}".format(err)); raise
        except subprocess.CalledProcessError as err:
            print("subprocess.CalledProcessError: {0} \n OmegaMaxEnt did not find the result \n".format(err))
        except:
            print("Unknown exception"); raise
        finally:
            if to_log and stdout: logging.info(stdout)


    def save_Aw_t(self, w_vec, Aw_t, fout_Aw_t=None):
        """ """
        if fout_Aw_t is None:
            fout_Aw_t = self.fout_Aw_t

        tmp = np.zeros((Aw_t.shape[0], Aw_t.shape[1] + 1))
        tmp[:, 0] = w_vec
        tmp[:, 1::] = Aw_t

        fmanip.backup_file(fout_Aw_t)
        np.savetxt(fout_Aw_t, tmp)

    def cleanup(self):
        """ """
        logging.info("\n\n End of acon \n\n")
