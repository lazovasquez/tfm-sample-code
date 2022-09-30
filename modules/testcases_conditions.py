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

# Simulation
if simulation == 2:
    discretization = 1
elif any([simulation == 3, simulation == 4, simulation == 5]):
    discretization = 2

# Geometry
# Inclination
betavector = np.deg2rad(betavectordeg)

# Condition inclination
if inclination == 1:
    beta = betavector[0]
elif inclination == 2:
    beta = betavector[1]
elif inclination == 3:
    beta = betavector[2]
elif inclination == 4:
    beta_1 = np.deg2rad(-1.5)
    beta_2 = np.deg2rad(1.5)
    # beta_1 = Constant(beta_1)
    # beta_2 = Constant(beta_2)
    # beta_fenics = Expression("x[0] < l/2 ? beta_1 : beta_2", degee = 1)
    # print("WARNING : is l = 14m + 23m ? according to Issa (2003)")

if show_data == 1:
    print("Inclination = ", inclination)

# Equations
# Body force
if dirichlet_type == 2:
    Fbody = 0
elif dirichlet_type == 1:
    Fbody = 74.225
    print("WARNING : is beta = 0? (to Sanderse, et al., 2017)")

# Effect of linear and nonlinear waves
if effect == 1:
    waves_effect = 1
elif effect == 2:
    waves_effect = 1e4

# Effect of viscosity
if viscous_terms == 1:
    visc = 1
elif viscous_terms == 2:
    visc = 0

# Cases
if case == 0:
    j_l = j_lcases[0]
    j_g = j_gcases[0]
    L = L_cases[0]
    # reference = 'sanderse'
    description_case = "horizontal stable (Sanderse et al., 2017)"
elif case == 1:
    j_l = j_lcases[1]
    j_g = j_gcases[1]
    L = L_cases[1]
    # reference = 'sanderse'
    description_case = "horizontal unstable (Sanderse et al., 2017)"
elif case == 2:
    j_l = j_lcases[2]
    j_g = j_gcases[2]
    L = L_cases[2]
    # reference = 'sanderse'
    description_case = "horizontal ill-posed (Sanderse et al., 2017)"
elif case == 3:
    j_l = j_lcases[3]
    j_g = j_gcases[3]
    L = L_cases[3]
    # reference = 'sanderse'
    description_case = "horizontal case B (Sanderse et al., 2017)"
elif case == 4:
    j_l = j_lcases[4]
    j_g = j_gcases[4]
    L = L_cases[4]
    # reference = 'sanderse'
    description_case = "horizontal case C (Sanderse et al., 2017)"
elif case == 5:
    # print ("INFO: sanderse case D.")
    j_l = j_lcases[5]
    j_g = j_gcases[5]
    L = L_cases[5]
    # reference = 'sanderse'
    description_case = "horizontal case D (Sanderse et al., 2017)"
elif case == 6:
    # print ("INFO: sanderse wave growth.")
    j_l = j_lcases[6]
    j_g = j_gcases[6]
    L = L_cases[6]
    # reference = 'sanderse'
    description_case = "horizontal wave growth case (Sanderse et al., 2017)"
elif case == 7:
    j_l = j_lcases[7]
    j_g = j_gcases[7]
    L = L_cases[7]
    # reference = 'ferrari'
    description_case = "horizontal case 1 (Ferrari, 2017)"
elif case == 8:
    # print ("INFO: slug flow Ferrari 2.")
    j_l = j_lcases[8]
    j_g = j_gcases[8]
    L = L_cases[8]
    # reference = 'ferrari'
    description_case = "horizontal case 2 (Ferrari, 2017)"
elif case == 9:
    # print ("INFO: Montini well-posed.")
    j_l = j_lcases[9]
    j_g = j_gcases[9]
    L = L_cases[9]
    # reference = 'montini'
    description_case = "horizontal well-posed (Montini, 2011)"
elif case == 10:
    # print ("INFO: case ill-posed (Montini, 2011) ")
    j_l = j_lcases[10]
    j_g = j_gcases[10]
    L = L_cases[10]
    # reference = 'montini'
    description_case = "horizontal ill-posed (Montini, 2011)"

if show_data == 1:
    print("INFO:", case, description_case)

# Time discretization
# Time discretization
if time_method == 1:
    a0 = 1.0
    a1 = -1.0
    a2 = 0.0
    theta = 1.0
    description_time = 'INFO: BDF1 first-order Backward differentiation formula.'
elif time_method == 2:
    a0 = 3/2
    a1 = -2.0
    a2 = 1/2
    theta = 1.0
    step_bdf2 = "first"
    description_time = 'INFO: BDF2 second-order Backward differentiation formula.'
elif time_method == 3:
    a0 = 1.0
    a1 = -1.0
    a2 = 0.0
    theta = 0.50
    description_time = 'INFO: CN Crank-Nicolson/trapezoidal.'

if show_data == 1:
    print("INFO:", time_method, description_time)

# (base) root@MacBook twofluidmodel # conda activate fenicsproject
# (fenicsproject) root@MacBook twofluidmodel # ./modules/constants_codes.py
