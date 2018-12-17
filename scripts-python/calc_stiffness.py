from mea.model import periodize_nambu, nambu
from mea.transport import stiffness
import numpy as np
from scipy.integrate import dblquad
import json
import os, glob

#stiffness.stiffness(fname="self_moy.dat")

def stiff_walk(fname="self_moy.dat"):
    """walk a directory and get the stiffness for all subdirectories """
    folderlist = list(map(os.path.abspath, os.listdir()))
    cwd = os.getcwd()

    stifflist = []
    for folder in folderlist:
        os.chdir(folder)
        os.chdir(glob.glob("Stats*")[0])
        result = stiffness.stiffness(fname)
        stifflist.append(result)
        os.chdir(cwd)
        with open("output_stiff_walk.dat", mode="a") as fout:
            for element in result:
                fout.write(str(element)); fout.write(" ")
            fout.write("\n")
    
    print("\nstifflist = ", stifflist)


def calc_stiffness(fname="self_moy.dat"):
    print(stiffness.stiffness(fname))


if __name__ == "__main__":
    
    try:
        stiff_walk()
    except:
        calc_stiffness()