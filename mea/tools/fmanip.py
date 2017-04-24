
import shutil
import os
import time



def backup_file(foutname):
    """ """
    assert(type(foutname) == str)
    
    if os.path.isfile(foutname):
        backup_name = foutname.split(".")[0] + \
                   "_old" + time.strftime("_%Y_%m_%d_%Hh%M") + "." + \
                   foutname.split(".")[1]
        
        shutil.move(foutname, backup_name)

def backup_folder(foldername):
    """ """
    assert(type(foldername) == str)
    if os.path.isdir(foldername):
        shutil.move(foldername, foldername + "_old" + time.strftime("_%Y_%m_%d_%Hh%M") )


def build_file_list(fin_name):
    """ """
    indata_list = []

    abs_dir = os.path.split(fin_name)[0] ; fname = os.path.split(fin_name)[1].split(".")[0]
    ext = os.path.split(fin_name)[1].split(".")[1]
    iteration = 0
    fname_tmp = os.path.join(abs_dir, fname + str(iteration) + "." + ext)
    while os.path.isfile(fname_tmp):
        with open(fname_tmp) as indata:
            indata_list.append(indata.readlines())
        iteration += 1
        fname_tmp = os.path.join(abs_dir, fname + str(iteration) + "." + ext)

    if indata_list == []:
        print("Invalid name format for file name: ") ; raise ValueError

    return indata_list


# def modify_file(fin_tomod, modifs, fout):
#         """ 
#         fin_tomod (str): the file as input to modify_file
#         modifs (list of strs): the lines to modify in fin_tomod
#         fout (str) : the file name to which write the modifications
#         """

#         with open(fin_tomod, mode="r") as input_tomod:
#             # 3.1) Modify the elements in the input_file
#             input_tomod_s = input_tomod.read()

#             for line in modifs:
#                 line_rstrip = line.rstrip()
#                 tmp = line.split(":")
#                 regx = ""
#                 for part in tmp[:-1]:
#                     regx += part + ":"
#                 regx = re.escape(regx)
#                 OME_input_s = re.sub(regx, line_strip, OME_input_s)

#             if iteration >= 1 and "real frequency grid file" not in " ".join(self.OME_input[iter_OME_input]):
#                 OME_input_s = re.sub(r"real frequency grid file:", "real frequency grid file:" + self.w_vec_file, OME_input_s)
#         #print(OME_input_s)

        # with open(fout, mode="w") as OME_output:
        #     OME_output.write(OME_input_s)

