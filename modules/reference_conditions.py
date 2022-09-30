#!/usr/bin/env python3

import argparse
import json
import math
import os
import pprint
import subprocess
import time
from ast import Num
from math import pi

import h5py
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.signal
import ufl
from fenics import *
from IPython.display import clear_output
from matplotlib import rc, style
from matplotlib.ticker import (AutoMinorLocator, FormatStrFormatter,
                               MultipleLocator)
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from scipy import linalg, matrix, sparse
from scipy.interpolate import interp1d
from scipy.linalg import eig, eigvals
from scipy.misc import derivative as dtv
from scipy.optimize import brenth, fsolve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

from modules.constants_codes import *
from modules.equations_functions import *
from modules.equations_matrices import *
from modules.fem_discretization import *
from modules.fully_discretized import *
from modules.initial_boundary_conditions import *
from modules.reference_conditions import *
from modules.semi_discretized import *
from modules.testcases_conditions import *
from modules.transient_solvers import *
from modules.variational_form import *

# Reference state: stratified smooth flow
if show_data == 1:
    print("Equation system :", system, ", case :",
          case, ", configuration :", inclination)
    print(" ")
    print(" ======================================================================")
    print(" ")

if any([inclination == 1, inclination == 2, inclination == 3]):
    # Pipe reference conditions
    ref, Re_ref = ref_state(j_l, j_g, var4_0, beta,
                            rho_l, p_factor, mu_l, mu_g, D)

    if show_data == 1:
        # Print reference conditions
        print("INFO: j_l = ", j_l)
        print("INFO: j_g = ", j_g)
        print(" ")
        print("INFO: var1_ref = ", ref[0], "[-]")
        print("INFO: var2_ref = ", ref[1], "[m/s]")
        print("INFO: var3_ref = ", ref[2], "[m/s]")
        print("INFO: var4_ref = ", ref[3], "[Pa]")
        print(" ")
        print("INFO: Rel_ref  = ", Re_ref[0], "[-]")
        print("INFO: Reg_ref  = ", Re_ref[1], "[-]")
        # print ("INFO: Regl_ref = ", Re_ref[2], "[-]")
        print(" ")

    # Linearized matrices
    Aeval, Beval, Ceval = linear_matrices_function(
        ref[0], ref[1], ref[2], ref[3], beta, D, system, rho_l, p_factor, mu_l, mu_g,
        dirichlet_type)

    # Define characteristics
    listreal1, listimag1, maxchar = characteristics_function(Aeval, Beval)

    if show_data == 1:
        # Print characteristics
        print("Re [lambda] =", listreal1)
        print("Im [lambda] =", listimag1)
        print(" ")
        print("max (Re [lambda]) =", maxchar)
        print(" ")

    # Define well-posedness
    if all([cond1 == 0 for cond1 in listimag1]):
        if show_data == 1:
            print("INFO: well-posed")
            print(" ")
        wp = "well-posed"
    else:
        if show_data == 1:
            print("INFO: ill-posed")
            print(" ")
        wp = "ill-posed"

    # Define stability
    listreal2, listimag2, _, vect2 = stability_function(Aeval, Beval, Ceval)

    # Print stability
    if wp == "well-posed":
        if all([cond2 > 0 for cond2 in listimag2]):
            if show_data == 1:
                print("Re [omega] =", listreal2)
                print("Im [omega] =", listimag2)
                print(" ")
                print("INFO: VKH stable")
        else:
            if show_data == 1:
                print("Re [omega] =", listreal2)
                print("Im [omega] =", listimag2)
                print(" ")
                print("INFO: VKH unstable")
                print(" ")
            local = np.where(np.asarray(listimag2) < 0)
            eigenvector = vect2[local]
            eigenvector_real = eigenvector.real
            eigenvector_imag = eigenvector.imag

            if show_data == 1:
                print(" ")
                print("INFO: Eigenvector")
                print(" ")
                print("r =", eigenvector)
                print("r_real =", eigenvector_real)
                print("r_imag =", eigenvector_imag)
                print(" ")

if inclination == 4:
    # SECTION 1
    # Pipe reference conditions
    ref = ref_state(j_l, j_g, var4_0, beta_1, rho_l, p_factor, mu_l, mu_g, D)

    # Print reference conditions
    print("SECTION 1")
    print(" ")
    print("INFO: var1_ref = ", ref[0], "[-]")
    print("INFO: var2_ref = ", ref[1], "[m/s]")
    print("INFO: var3_ref = ", ref[2], "[m/s]")
    print("INFO: var4_ref = ", ref[3], "[Pa]")
    print(" ")
    # print ("INFO: Rel_ref  = ", Re_ref[0], "[-]")
    # print ("INFO: Reg_ref  = ", Re_ref[1], "[-]")
    # print ("INFO: Regl_ref = ", Re_ref[2], "[-]")
    # print (" ")

    # Linearized matrices
    Aeval_1, Beval_1, Ceval_1 = linear_matrices_function(
        ref[0], ref[1], ref[2], ref[3], beta_1, D, system, rho_l, p_factor, mu_l, mu_g, dirichlet_type)

    # Characteristics
    listreal1_1, listimag1_1, maxchar_1 = characteristics_function(
        Aeval_1, Beval_1)

    # Print characteristics
    print("Re [lambda] =", listreal1_1)
    print("Im [lambda] =", listimag1_1)
    print(" ")
    print("max (Re [lambda]) =", maxchar_1)
    print(" ")

    # Define well-posedness
    if all([cond3 == 0 for cond3 in listimag1_1]):
        print("INFO: well-posed")
        wp_1 = "well-posed"
    else:
        print("INFO: ill-posed")
        wp_1 = "ill-posed"

    print(" ")

    # Stability
    listreal2_1, listimag2_1, _, vect2_1 = stability_function(
        Aeval_1, Beval_1, Ceval_1)

    # print stability
    if wp_1 == "well-posed":
        if all([cond4 > 0 for cond4 in listimag2_1]):
            print("Re [omega] =", listreal2_1)
            print("Im [omega] =", listimag2_1)
            print(" ")
            print("INFO: VKH stable")
        else:
            print("Re [omega] =", listreal2_1)
            print("Im [omega] =", listimag2_1)
            print(" ")
            print("INFO: VKH unstable")
            print(" ")
            local_1 = np.where(np.asarray(listimag2_1) < 0)
            eigenvector_1 = vect2_1[local_1]
            print("r =", eigenvector_1)

    # Define alpha for section 2
    alpha_section1 = ref[0]

    # SECTION 2
    # Pipe reference conditions
    # ref = ref_state (u1_section1*ref[0]*alpha_section1, u2_section1*(1 - alpha_section1), var4_0, beta_2, rho_l, p_factor, mu_l, mu_g, D)
    del (ref, j_l)
    ref = ref_state_jl(alpha_section1, j_g, var4_0, beta_2,
                       rho_l, p_factor, mu_l, mu_g, D)

    # Print reference conditions
    print(" ")
    print(" ======================================================================")
    print(" ")
    print("SECTION 2")
    print(" ")
    print("INFO: var1_ref = ", ref[0], "[-]")
    print("INFO: var2_ref = ", ref[1], "[m/s]")
    print("INFO: var3_ref = ", ref[2], "[m/s]")
    print("INFO: var4_ref = ", ref[3], "[Pa]")
    print(" ")
    # print ("INFO: Rel_ref  = ", Re_ref[0], "[-]")
    # print ("INFO: Reg_ref  = ", Re_ref[1], "[-]")
    # print ("INFO: Regl_ref = ", Re_ref[2], "[-]")
    # print (" ")

    # Linearized matrices
    Aeval_2, Beval_2, Ceval_2 = linear_matrices_function(
        ref[0], ref[1], ref[2], ref[3], beta_2, D, system, rho_l, p_factor, mu_l, mu_g, dirichlet_type)

    # Characteristics
    listreal1_2, listimag1_2, maxchar_2 = characteristics_function(
        Aeval_2, Beval_2)

    # Print characteristics
    print("Re [lambda] =", listreal1_2)
    print("Im [lambda] =", listimag1_2)
    print(" ")
    print("max (Re [lambda]) =", maxchar_2)
    print(" ")

    # Define well-posedness
    if all([cond5 == 0 for cond5 in listimag1_2]):
        print("INFO: well-posed")
        wp_2 = "well-posed"
    else:
        print("INFO: ill-posed")
        wp_2 = "ill-posed"

    print(" ")

    # Stability
    listreal2_2, listimag2_2, _, vect2_2 = stability_function(
        Aeval_2, Beval_2, Ceval_2)

    if wp_2 == "well-posed":
        if all([cond6 > 0 for cond6 in listimag2_2]):
            print("Re [omega] =", listreal2_2)
            print("Im [omega] =", listimag2_2)
            print(" ")
            print("INFO: VKH stable")
        else:
            print("Re [omega] =", listreal2_2)
            print("Im [omega] =", listimag2_2)
            print(" ")
            print("INFO: VKH unstable")
            print(" ")
            local_2 = np.where(np.asarray(listimag2_2) < 0)
            eigenvector_2 = vect2_2[local_2]
            print("r =", eigenvector_2)

    print(" ")
    print(" ======================================================================")
    print(" ")
    print("max (Re [lambda]) =", max(maxchar_1, maxchar_2))

# Define reference for system 3
if system == 3:
    ref1__ = ref[0]*rho_l
    ref2__ = (1 - ref[0])*(p_factor*ref[3] + DOLFIN_EPS)
    ref3__ = ref1__*ref[1]
    ref4__ = (1 - ref[0])*(p_factor*ref[3] + DOLFIN_EPS)*ref[2]

    # Define reference vector
    ref = np.array([ref1__, ref2__, ref3__, ref4__])

    print(" ")
    print("INFO: cvar1_ref = ", ref[0])
    print("INFO: cvar2_ref = ", ref[1])
    print("INFO: cvar3_ref = ", ref[2])
    print("INFO: cvar4_ref = ", ref[3])

# > Stratified wavy plot
# varstrings = [r'$\alpha_l$', '$u_l$', '$u_g$', '$p_i$']
# waves = ["acoustic1", "convective1", "convective2", "acoustic2"]

# omegatest = np.array ([-1758.05, 4.27, 8.48, 1931.47])
# rtest = np.array ([ 1e-6 , 7.005e-7, 2.497e-5, -3.619e-4 ])

# i=0
# for i in range(4):
#     print(waves[i], "wave")
#     print("eigenvector =", rtest[i])
#     fig   = plt.figure ()
#     ax    = fig.gca (projection='3d')

#     x     = np.zeros (0, L, n_wavy)
#     y     = np.zeros (0, T_in, n_wavy)
#     z     = ref[i] + rtest[i]*(np.exp ( 1j*(omegatest[2]*y-wavenumber_fourier*x))).real
#     # z     = ref[i] + rtest[i]*(np.sin ((omegatest[2]*y-wavenumber_fourier*x)))               # 2nd alternative

#     ax.plot(x, y, z, color = listcolor [0])
#     matplotlib.rcParams ['legend.fontsize'] = 10

#     plt.rcParams ['figure.figsize'] = mapsize
#     # leg1  = ax.legend (loc = 'best', frameon = True, fontsize = 14);
#     plt.grid (True, which = "both")

#     ax.set_xlabel('s[m]', fontsize = 16)
#     ax.set_ylabel('t[s]', fontsize = 16)
#     ax.set_zlabel(varstrings[i], fontsize = 16)

#     ax.ticklabel_format(useOffset=False) #<<<<<<<<<<<<<<<<<<

#     sup_lim = ref[i] + rtest[i]
#     inf_lim = ref[i] - rtest[i]

#     ax.set_xlim ((0, L))
#     ax.set_ylim ((0, T_in))
#     ax.set_zlim((inf_lim, sup_lim))

#     plt.show()
#     i+=1


# (base) root@MacBook twofluidmodel # conda activate fenicsproject
# (fenicsproject) root@MacBook twofluidmodel # ./modules/reference_conditions.py
