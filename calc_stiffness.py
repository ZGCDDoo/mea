from mea.model import periodize_nambu, nambu
from mea.transport import stiffness
import numpy as np
from scipy.integrate import dblquad
import json


stiffness.stiffness(fname="self_moy.dat")


