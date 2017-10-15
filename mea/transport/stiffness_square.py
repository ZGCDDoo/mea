import nambu_square
import numpy as np
from scipy.integrate import dblquad, simps


#For Olivier and qcm

#Copyright Charles-David Hebert
#MIT Licencse, use it as you see fit, but please give retributions to the author.

def stiffness(model_sc):

    ss = model_sc.shape[0]
    stiffness_green_arr = np.zeros(ss)
    stiffness_cum_arr = np.zeros(ss)
    stiffness_trace_arr = np.zeros(ss)

    Y1 = model_sc.Y1Limit
    Y2 = model_sc.Y2Limit

    N_c = 4.0
    for ii in range(ss):
        stiffness_green_arr[ii] = 2.0/(2.0*np.pi)**2*dblquad(model_sc.stiffness, -np.pi, np.pi, Y1, Y2, args=(ii,) )[0]
        stiffness_cum_arr[ii] = 2.0/(2.0*np.pi)**2*dblquad(model_sc.stiffness_cum, -np.pi, np.pi, Y1, Y2, args=(ii,) )[0]
        stiffness_trace_arr[ii] = 2.0*N_c/(2.0*np.pi)**2*dblquad(model_sc.stiffness_trace, -np.pi/2.0, np.pi/2.0, 
                                                                lambda x: -np.pi/2.0, lambda x: np.pi/2.0, args=(ii,) )[0]


    stiffness_green = 1.0/(2.0*np.pi)*simps(stiffness_green_arr, model_sc.z_vec)
    stiffness_cum = 1.0/(2.0*np.pi)*simps(stiffness_cum_arr, model_sc.z_vec)
    stiffness_trace = 1.0/(2.0*np.pi)*simps(stiffness_trace_arr, model_sc.z_vec)

    np.savetxt("stiffness.dat", np.array([[stiffness_green, stiffness_cum, stiffness_trace]]))
    return (stiffness_green, stiffness_cum, stiffness_trace)


