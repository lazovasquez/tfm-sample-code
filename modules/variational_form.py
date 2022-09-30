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

# Linear simulations
# def variational_form_linear(Aeval, Beval, Ceval):
if any([simulation == 2, simulation == 3, simulation == 4]):
    # Amat coefficient
    A11_ref = Constant(Aeval[0][0])
    A12_ref = Constant(Aeval[0][1])
    A13_ref = Constant(Aeval[0][2])
    A14_ref = Constant(Aeval[0][3])

    A21_ref = Constant(Aeval[1][0])
    A22_ref = Constant(Aeval[1][1])
    A23_ref = Constant(Aeval[1][2])
    A24_ref = Constant(Aeval[1][3])

    A31_ref = Constant(Aeval[2][0])
    A32_ref = Constant(Aeval[2][1])
    A33_ref = Constant(Aeval[2][2])
    A34_ref = Constant(Aeval[2][3])

    A41_ref = Constant(Aeval[3][0])
    A42_ref = Constant(Aeval[3][1])
    A43_ref = Constant(Aeval[3][2])
    A44_ref = Constant(Aeval[3][3])

    # Bmat coefficient
    B11_ref = Constant(Beval[0][0])
    B12_ref = Constant(Beval[0][1])
    B13_ref = Constant(Beval[0][2])
    B14_ref = Constant(Beval[0][3])

    B21_ref = Constant(Beval[1][0])
    B22_ref = Constant(Beval[1][1])
    B23_ref = Constant(Beval[1][2])
    B24_ref = Constant(Beval[1][3])

    B31_ref = Constant(Beval[2][0])
    B32_ref = Constant(Beval[2][1])
    B33_ref = Constant(Beval[2][2])
    B34_ref = Constant(Beval[2][3])

    B41_ref = Constant(Beval[3][0])
    B42_ref = Constant(Beval[3][1])
    B43_ref = Constant(Beval[3][2])
    B44_ref = Constant(Beval[3][3])

    # Bmat
    Bm1 = (dot(B11_ref*Dx(variable1, 0), v1)
           + dot(B12_ref*Dx(variable2, 0), v1)
           + dot(B13_ref*Dx(variable3, 0), v1)
           + dot(B14_ref*Dx(variable4, 0), v1))*dx

    Bm2 = (dot(B21_ref*Dx(variable1, 0), v2)
           + dot(B22_ref*Dx(variable2, 0), v2)
           + dot(B23_ref*Dx(variable3, 0), v2)
           + dot(B24_ref*Dx(variable4, 0), v2))*dx

    Bm3 = (dot(B31_ref*Dx(variable1, 0), v3)
           + dot(B32_ref*Dx(variable2, 0), v3)
           + dot(B33_ref*Dx(variable3, 0), v3)
           + dot(B34_ref*Dx(variable4, 0), v3))*dx

    Bm4 = (dot(B41_ref*Dx(variable1, 0), v4)
           + dot(B42_ref*Dx(variable2, 0), v4)
           + dot(B43_ref*Dx(variable3, 0), v4)
           + dot(B44_ref*Dx(variable4, 0), v4))*dx

    # Cmat coefficient
    C11_ref = Constant(Ceval[0][0])
    C12_ref = Constant(Ceval[0][1])
    C13_ref = Constant(Ceval[0][2])
    C14_ref = Constant(Ceval[0][3])

    C21_ref = Constant(Ceval[1][0])
    C22_ref = Constant(Ceval[1][1])
    C23_ref = Constant(Ceval[1][2])
    C24_ref = Constant(Ceval[1][3])

    C31_ref = Constant(Ceval[2][0])
    C32_ref = Constant(Ceval[2][1])
    C33_ref = Constant(Ceval[2][2])
    C34_ref = Constant(Ceval[2][3])

    C41_ref = Constant(Ceval[3][0])
    C42_ref = Constant(Ceval[3][1])
    C43_ref = Constant(Ceval[3][2])
    C44_ref = Constant(Ceval[3][3])

    # Cmat_lin
    Cm1_lin = (dot(C11_ref*variable1, v1)
               + dot(C12_ref*variable2, v1)
               + dot(C13_ref*variable3, v1)
               + dot(C14_ref*variable4, v1))*dx

    Cm2_lin = (dot(C21_ref*variable1, v2)
               + dot(C22_ref*variable2, v2)
               + dot(C23_ref*variable3, v2)
               + dot(C24_ref*variable4, v2))*dx

    Cm3_lin = (dot(C31_ref*variable1, v3)
               + dot(C32_ref*variable2, v3)
               + dot(C33_ref*variable3, v3)
               + dot(C34_ref*variable4, v3))*dx

    Cm4_lin = (dot(C41_ref*variable1, v4)
               + dot(C42_ref*variable2, v4)
               + dot(C43_ref*variable3, v4)
               + dot(C44_ref*variable4, v4))*dx

    # Variational form
    Bm = Bm1 + Bm2 + Bm3 + Bm4
    Cm = Cm1_lin + Cm2_lin + Cm3_lin + Cm4_lin

    if any([discretization == 2, discretization == 3, discretization == 4]):
        # if any([time_method == 'BDF1', time_method == 'CN']):
        # Bmat_n
        Bm1_n = (inner(B11_ref*Dx(variable1_n, 0), v1)
                 + inner(B12_ref*Dx(variable2_n, 0), v1)
                 + inner(B13_ref*Dx(variable3_n, 0), v1)
                 + inner(B14_ref*Dx(variable4_n, 0), v1))*dx

        Bm2_n = (inner(B21_ref*Dx(variable1_n, 0), v2)
                 + inner(B22_ref*Dx(variable2_n, 0), v2)
                 + inner(B23_ref*Dx(variable3_n, 0), v2)
                 + inner(B24_ref*Dx(variable4_n, 0), v2))*dx

        Bm3_n = (inner(B31_ref*Dx(variable1_n, 0), v3)
                 + inner(B32_ref*Dx(variable2_n, 0), v3)
                 + inner(B33_ref*Dx(variable3_n, 0), v3)
                 + inner(B34_ref*Dx(variable4_n, 0), v3))*dx

        Bm4_n = (inner(B41_ref*Dx(variable1_n, 0), v4)
                 + inner(B42_ref*Dx(variable2_n, 0), v4)
                 + inner(B43_ref*Dx(variable3_n, 0), v4)
                 + inner(B44_ref*Dx(variable4_n, 0), v4))*dx

        # Cmat_lin_n
        Cm1_lin_n = (inner(C11_ref*variable1_n, v1)
                     + inner(C12_ref*variable2_n, v1)
                     + inner(C13_ref*variable3_n, v1)
                     + inner(C14_ref*variable4_n, v1))*dx

        Cm2_lin_n = (inner(C21_ref*variable1_n, v2)
                     + inner(C22_ref*variable2_n, v2)
                     + inner(C23_ref*variable3_n, v2)
                     + inner(C24_ref*variable4_n, v2))*dx

        Cm3_lin_n = (inner(C31_ref*variable1_n, v3)
                     + inner(C32_ref*variable2_n, v3)
                     + inner(C33_ref*variable3_n, v3)
                     + inner(C34_ref*variable4_n, v3))*dx

        Cm4_lin_n = (inner(C41_ref*variable1_n, v4)
                     + inner(C42_ref*variable2_n, v4)
                     + inner(C43_ref*variable3_n, v4)
                     + inner(C44_ref*variable4_n, v4))*dx

        # Variational form
        Bm_n = Bm1_n + Bm2_n + Bm3_n + Bm4_n
        Cm_n = Cm1_lin_n + Cm2_lin_n + Cm3_lin_n + Cm4_lin_n

        if time_method == 2:
            # Bmat_past
            Bm1_past = (inner(B11_ref*Dx(variable1_past, 0), v1)
                        + inner(B12_ref*Dx(variable2_past, 0), v1)
                        + inner(B13_ref*Dx(variable3_past, 0), v1)
                        + inner(B14_ref*Dx(variable4_past, 0), v1))*dx

            Bm2_past = (inner(B21_ref*Dx(variable1_past, 0), v2)
                        + inner(B22_ref*Dx(variable2_past, 0), v2)
                        + inner(B23_ref*Dx(variable3_past, 0), v2)
                        + inner(B24_ref*Dx(variable4_past, 0), v2))*dx

            Bm3_past = (inner(B31_ref*Dx(variable1_past, 0), v3)
                        + inner(B32_ref*Dx(variable2_past, 0), v3)
                        + inner(B33_ref*Dx(variable3_past, 0), v3)
                        + inner(B34_ref*Dx(variable4_past, 0), v3))*dx

            Bm4_past = (inner(B41_ref*Dx(variable1_past, 0), v4)
                        + inner(B42_ref*Dx(variable2_past, 0), v4)
                        + inner(B43_ref*Dx(variable3_past, 0), v4)
                        + inner(B44_ref*Dx(variable4_past, 0), v4))*dx

            # Cmat_lin_past
            Cm1_lin_past = (inner(C11_ref*variable1_past, v1)
                            + inner(C12_ref*variable2_past, v1)
                            + inner(C13_ref*variable3_past, v1)
                            + inner(C14_ref*variable4_past, v1))*dx

            Cm2_lin_past = (inner(C21_ref*variable1_past, v2)
                            + inner(C22_ref*variable2_past, v2)
                            + inner(C23_ref*variable3_past, v2)
                            + inner(C24_ref*variable4_past, v2))*dx

            Cm3_lin_past = (inner(C31_ref*variable1_past, v3)
                            + inner(C32_ref*variable2_past, v3)
                            + inner(C33_ref*variable3_past, v3)
                            + inner(C34_ref*variable4_past, v3))*dx

            Cm4_lin_past = (inner(C41_ref*variable1_past, v4)
                            + inner(C42_ref*variable2_past, v4)
                            + inner(C43_ref*variable3_past, v4)
                            + inner(C44_ref*variable4_past, v4))*dx

            # Variational form
            Bm_past = Bm1_past + Bm2_past + Bm3_past + Bm4_past
            Cm_past = Cm1_lin_past + Cm2_lin_past + Cm3_lin_past + Cm4_lin_past
    # return

# Nonlinear simulations

if simulation == 5:
    # if any([simulation == "stiffness" , simulation == "linear_simulation"]):
    # Define constants

    # D = Constant(D)
    # A = Constant(A)
    # g = Constant(g)
    # rho_l = Constant(rho_l)
    # pi = Constant(pi)
    # p_factor = Constant(p_factor)
    # mu_l = Constant(mu_l)
    # mu_g = Constant(mu_g)
    # j_g = Constant(j_g)

    # Bmat
    Bm1 = (dot(variable2*Dx(variable1, 0), v1)
           + dot(variable1*Dx(variable2, 0), v1)
           + dot(Constant(0.0)*Dx(variable3, 0), v1)
           + dot(Constant(0.0)*Dx(variable4, 0), v1))*dx

    Bm2 = (dot(Dc_fenics(variable1)*Dx(variable1, 0), v2)
           + dot(variable2*Dx(variable2, 0), v2)
           + dot(Constant(0.0)*Dx(variable3, 0), v2)
           + dot(1/rho_l*Dx(variable4, 0), v2))*dx

    Bm3 = (dot(Dc_fenics(variable1)*Dx(variable1, 0), v3)
           + dot(Constant(0.0)*Dx(variable2, 0), v3)
           + dot(variable3*Dx(variable3, 0), v3)
           + dot(1/rho_g(variable4)*Dx(variable4, 0), v3))*dx

    Bm4 = (dot((-variable2 + variable3)*variable4/(-1 + variable1)*Dx(variable1, 0), v4)
           + dot(variable1*variable4/(1 - variable1)*Dx(variable2, 0), v4)
           + dot(variable4*Dx(variable3, 0), v4)
           + dot(variable3*Dx(variable4, 0), v4))*dx

    # # Bmat
    # Bm1 = ( inner(B11_fenics(variable1, variable2, variable3, variable4)*Dx(variable1, 0), v1) \
    #         + inner(B12_fenics(variable1, variable2, variable3, variable4)*Dx(variable2, 0), v1) \
    #         + inner(B13_fenics(variable1, variable2, variable3, variable4)*Dx(variable3, 0), v1) \
    #         + inner(B14_fenics(variable1, variable2, variable3, variable4)*Dx(variable4, 0), v1) )*dx

    # Bm2 = ( inner(B21_fenics(variable1, variable2, variable3, variable4)*Dx(variable1, 0), v2) \
    #         + inner(B22_fenics(variable1, variable2, variable3, variable4)*Dx(variable2, 0), v2) \
    #         + inner(B23_fenics(variable1, variable2, variable3, variable4)*Dx(variable3, 0), v2) \
    #         + inner(B24_fenics(variable1, variable2, variable3, variable4)*Dx(variable4, 0), v2) )*dx

    # Bm3 = ( inner(B31_fenics(variable1, variable2, variable3, variable4)*Dx(variable1, 0), v3) \
    #         + inner(B32_fenics(variable1, variable2, variable3, variable4)*Dx(variable2, 0), v3) \
    #         + inner(B33_fenics(variable1, variable2, variable3, variable4)*Dx(variable3, 0), v3) \
    #         + inner(B34_fenics(variable1, variable2, variable3, variable4)*Dx(variable4, 0), v3) )*dx

    # Bm4 = ( inner(B41_fenics(variable1, variable2, variable3, variable4)*Dx(variable1, 0), v4) \
    #         + inner(B42_fenics(variable1, variable2, variable3, variable4)*Dx(variable2, 0), v4) \
    #         + inner(B43_fenics(variable1, variable2, variable3, variable4)*Dx(variable3, 0), v4) \
    #         + inner(B44_fenics(variable1, variable2, variable3, variable4)*Dx(variable4, 0), v4) )*dx

    # Cmat
    Cm1 = ((Constant(0.0)))*v1*dx

    Cm2 = ((sin(beta)*g*rho_l*variable1 +
           tau_gl_fenics(variable1, variable2, variable3, variable4)*P_gl_fenics(variable1)/A -
           tau_lw(variable1, variable2)*P_lw(variable1)/A)/(rho_l*(variable1+DOLFIN_EPS))
           )*v2*dx

    Cm3 = ((-sin(beta)*g +
            (tau_gl_fenics(variable1, variable2, variable3, variable4)*P_gl_fenics(variable1)/A + tau_gw_fenics(variable1, variable3, variable4)*P_gw(variable1)/A)/(rho_g(variable4)*(-1 + variable1)))
           )*v3*dx

    Cm4 = ((Constant(0.0)))*v4*dx

    # Cmat
    # Cm1 =( inner(Cmat1_fenics(variable1, variable2, variable3, variable4), v1) )*dx
    # Cm2 =( inner(Cmat2_fenics(variable1, variable2, variable3, variable4), v2) )*dx
    # Cm3 =( inner(Cmat3_fenics(variable1, variable2, variable3, variable4), v3) )*dx
    # Cm4 =( inner(Cmat4_fenics(variable1, variable2, variable3, variable4), v4) )*dx

    # Variational form
    Bm = Bm1 + Bm2 + Bm3 + Bm4
    Cm = Cm1 + Cm2 + Cm3 + Cm4

    if discretization == 4:
        # Define constants
        dt = Constant(dt)

        # Bmat_n
        Bm1_n = (inner(B11_fenics(variable1_n, variable2_n, variable3_n, variable4_n)*Dx(variable1_n, 0), v1)
                 + inner(B12_fenics(variable1_n, variable2_n,
                         variable3_n, variable4_n)*Dx(variable2_n, 0), v1)
                 + inner(B13_fenics(variable1_n, variable2_n,
                         variable3_n, variable4_n)*Dx(variable3_n, 0), v1)
                 + inner(B14_fenics(variable1_n, variable2_n, variable3_n, variable4_n)*Dx(variable4_n, 0), v1))*dx

        Bm2_n = (inner(B21_fenics(variable1_n, variable2_n, variable3_n, variable4_n)*Dx(variable1_n, 0), v2)
                 + inner(B22_fenics(variable1_n, variable2_n,
                         variable3_n, variable4_n)*Dx(variable2_n, 0), v2)
                 + inner(B23_fenics(variable1_n, variable2_n,
                         variable3_n, variable4_n)*Dx(variable3_n, 0), v2)
                 + inner(B24_fenics(variable1_n, variable2_n, variable3_n, variable4_n)*Dx(variable4_n, 0), v2))*dx

        Bm3_n = (inner(B31_fenics(variable1_n, variable2_n, variable3_n, variable4_n)*Dx(variable1_n, 0), v3)
                 + inner(B32_fenics(variable1_n, variable2_n,
                         variable3_n, variable4_n)*Dx(variable2_n, 0), v3)
                 + inner(B33_fenics(variable1_n, variable2_n,
                         variable3_n, variable4_n)*Dx(variable3_n, 0), v3)
                 + inner(B34_fenics(variable1_n, variable2_n, variable3_n, variable4_n)*Dx(variable4_n, 0), v3))*dx

        Bm4_n = (inner(B41_fenics(variable1_n, variable2_n, variable3_n, variable4_n)*Dx(variable1_n, 0), v4)
                 + inner(B42_fenics(variable1_n, variable2_n,
                         variable3_n, variable4_n)*Dx(variable2_n, 0), v4)
                 + inner(B43_fenics(variable1_n, variable2_n,
                         variable3_n, variable4_n)*Dx(variable3_n, 0), v4)
                 + inner(B44_fenics(variable1_n, variable2_n, variable3_n, variable4_n)*Dx(variable4_n, 0), v4))*dx

        # Cmat_n
        Cm1_n = (inner(Cmat1_fenics(variable1_n, variable2_n,
                 variable3_n, variable4_n), v1))*dx
        Cm2_n = (inner(Cmat2_fenics(variable1_n, variable2_n,
                 variable3_n, variable4_n), v2))*dx
        Cm3_n = (inner(Cmat3_fenics(variable1_n, variable2_n,
                 variable3_n, variable4_n), v3))*dx
        Cm4_n = (inner(Cmat4_fenics(variable1_n, variable2_n,
                 variable3_n, variable4_n), v4))*dx

        # Variational form
        Bm_n = Bm1_n + Bm2_n + Bm3_n + Bm4_n
        Cm_n = Cm1_n + Cm2_n + Cm3_n + Cm4_n

        if time_method == 2:
            # Bmat_past
            Bm1_past = (inner(B11(variable1_past, variable2_past, variable3_past, variable4_past)*Dx(variable1_past, 0), v1)
                        + inner(B12(variable1_past, variable2_past, variable3_past,
                                variable4_past)*Dx(variable2_past, 0), v1)
                        + inner(B13(variable1_past, variable2_past, variable3_past,
                                variable4_past)*Dx(variable3_past, 0), v1)
                        + inner(B14(variable1_past, variable2_past, variable3_past, variable4_past)*Dx(variable4_past, 0), v1))*dx

            Bm2_past = (inner(B21_fenics(variable1_past, variable2_past, variable3_past, variable4_past)*Dx(variable1_past, 0), v2)
                        + inner(B22(variable1_past, variable2_past, variable3_past,
                                variable4_past)*Dx(variable2_past, 0), v2)
                        + inner(B23(variable1_past, variable2_past, variable3_past,
                                variable4_past)*Dx(variable3_past, 0), v2)
                        + inner(B24(variable1_past, variable2_past, variable3_past, variable4_past)*Dx(variable4_past, 0), v2))*dx

            Bm3_past = (inner(B31_fenics(variable1_past, variable2_past, variable3_past, variable4_past)*Dx(variable1_past, 0), v3)
                        + inner(B32(variable1_past, variable2_past, variable3_past,
                                variable4_past)*Dx(variable2_past, 0), v3)
                        + inner(B33(variable1_past, variable2_past, variable3_past,
                                variable4_past)*Dx(variable3_past, 0), v3)
                        + inner(B34(variable1_past, variable2_past, variable3_past, variable4_past)*Dx(variable4_past, 0), v3))*dx

            Bm4_past = (inner(B41(variable1_past, variable2_past, variable3_past, variable4_past)*Dx(variable1_past, 0), v4)
                        + inner(B42(variable1_past, variable2_past, variable3_past,
                                variable4_past)*Dx(variable2_past, 0), v4)
                        + inner(B43(variable1_past, variable2_past, variable3_past,
                                variable4_past)*Dx(variable3_past, 0), v4)
                        + inner(B44(variable1_past, variable2_past, variable3_past, variable4_past)*Dx(variable4_past, 0), v4))*dx

            # Cm_past
            Cm1_past = (inner(Cmat1_fenics(
                variable1_past, variable2_past, variable3_past, variable4_past), v1))*dx
            Cm2_past = (inner(Cmat2_fenics(
                variable1_past, variable2_past, variable3_past, variable4_past), v2))*dx
            Cm3_past = (inner(Cmat3_fenics(
                variable1_past, variable2_past, variable3_past, variable4_past), v3))*dx
            Cm4_past = (inner(Cmat4_fenics(
                variable1_past, variable2_past, variable3_past, variable4_past), v4))*dx

            # Variational form
            Bm_past = Bm1_past + Bm2_past + Bm3_past + Bm4_past
            Cm_past = Cm1_past + Cm2_past + Cm3_past + Cm4_past

# (base) root@MacBook twofluidmodel # conda activate fenicsproject
# (fenicsproject) root@MacBook twofluidmodel # ./modules/variational_form.py
