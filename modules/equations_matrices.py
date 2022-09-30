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

# # 1. Equation systems for linear stability
# W = (alpha_l, u_l, u_g, p_i) with A != I

if system == 1:
    # Amat
    def A11(var1, var2, var3, var4):
        return rho_l

    def A12(var1, var2, var3, var4):
        return 0

    def A13(var1, var2, var3, var4):
        return 0

    def A14(var1, var2, var3, var4):
        return 0

    def A21(var1, var2, var3, var4):
        return -rho_g(var4)

    def A22(var1, var2, var3, var4):
        return 0

    def A23(var1, var2, var3, var4):
        return 0

    def A24(var1, var2, var3, var4):
        return p_factor*(1 - var1)

    def A31(var1, var2, var3, var4):
        return rho_l*var2

    def A32(var1, var2, var3, var4):
        return rho_l*var1

    def A33(var1, var2, var3, var4):
        return 0

    def A34(var1, var2, var3, var4):
        return 0

    def A41(var1, var2, var3, var4):
        return -rho_g(var4)*var3

    def A42(var1, var2, var3, var4):
        return 0

    def A43(var1, var2, var3, var4):
        return rho_g(var4)*(1 - var1)

    def A44(var1, var2, var3, var4):
        return p_factor*(1 - var1)*var3

    # Bmat
    def B11(var1, var2, var3, var4):
        return rho_l*var2

    def B12(var1, var2, var3, var4):
        return rho_l*var1

    def B13(var1, var2, var3, var4):
        return 0

    def B14(var1, var2, var3, var4):
        return 0

    def B21(var1, var2, var3, var4):
        return -rho_g(var4)*var3

    def B22(var1, var2, var3, var4):
        return 0

    def B23(var1, var2, var3, var4):
        return rho_g(var4)*(1 - var1)

    def B24(var1, var2, var3, var4):
        return p_factor*(1 - var1)*var3

    def B31(var1, var2, var3, var4):
        return rho_l*var2**2 + rho_l*Dc(var1)*var1

    def B32(var1, var2, var3, var4):
        return 2*rho_l*var1*var2

    def B33(var1, var2, var3, var4):
        return 0

    def B34(var1, var2, var3, var4):
        return var1

    def B41(var1, var2, var3, var4):
        return -rho_g(var4)*var3**2 + rho_g(var4)*Dc(var1)*(1 - var1)

    def B42(var1, var2, var3, var4):
        return 0

    def B43(var1, var2, var3, var4):
        return rho_g(var4)*2*(1 - var1)*var3

    def B44(var1, var2, var3, var4):
        return p_factor*(1 - var1)*var3**2 + (1 - var1)

    # Cmat
    def Cmat1(var1, var2, var3, var4):
        return 0

    def Cmat2(var1, var2, var3, var4):
        return 0

    def Cmat3(var1, var2, var3, var4):
        return -rho_l*g*np.sin(beta)*var1 - tau_lw(var1, var2)*P_lw(var1)/A + tau_gl(var1, var2, var3, var4)*P_gl(var1)/A + Fbody*var1

    def Cmat4(var1, var2, var3, var4):
        return -rho_g(var4)*g*np.sin(beta)*(1 - var1) - tau_gw(var1, var3, var4)*P_gw(var1)/A - tau_gl(var1, var2, var3, var4)*P_gl(var1)/A + Fbody*(1 - var1)

    # Dmat
    def Dmat(var1, var4, Gamma_l, Gamma_g, nu_l, nu_g):
        Dmat = [[rho_l*Gamma_l, 0, 0, 0],

                [-rho_g(var4)*Gamma_g, 0, 0, 0],

                [0, var1*nu_l, 0, 0],

                [0, 0, nu_g*(1 - var1), 0]]
        return Dmat

# W : (alpha_l, u_l, u_g, p_i) with A = I

if system == 2:
    # Amat
    def A11(var1, var2, var3, var4):
        return 1

    def A12(var1, var2, var3, var4):
        return 0

    def A13(var1, var2, var3, var4):
        return 0

    def A14(var1, var2, var3, var4):
        return 0

    def A21(var1, var2, var3, var4):
        return 0

    def A22(var1, var2, var3, var4):
        return 1

    def A23(var1, var2, var3, var4):
        return 0

    def A24(var1, var2, var3, var4):
        return 0

    def A31(var1, var2, var3, var4):
        return 0

    def A32(var1, var2, var3, var4):
        return 0

    def A33(var1, var2, var3, var4):
        return 1

    def A34(var1, var2, var3, var4):
        return 0

    def A41(var1, var2, var3, var4):
        return 0

    def A42(var1, var2, var3, var4):
        return 0

    def A43(var1, var2, var3, var4):
        return 0

    def A44(var1, var2, var3, var4):
        return 1

    # Bmat
    def B11(var1, var2, var3, var4):
        return var2

    def B12(var1, var2, var3, var4):
        return var1

    def B13(var1, var2, var3, var4):
        return 0

    def B14(var1, var2, var3, var4):
        return 0

    def B21(var1, var2, var3, var4):
        return Dc(var1)

    def B22(var1, var2, var3, var4):
        return var2

    def B23(var1, var2, var3, var4):
        return 0

    def B24(var1, var2, var3, var4):
        return 1/(rho_l)

    def B31(var1, var2, var3, var4):
        return Dc(var1)

    def B32(var1, var2, var3, var4):
        return 0

    def B33(var1, var2, var3, var4):
        return var3

    def B34(var1, var2, var3, var4):
        return 1/rho_g(var4)

    def B41(var1, var2, var3, var4):
        return (-var2 + var3)*var4/(-1 + var1)

    def B42(var1, var2, var3, var4):
        return var1*var4/(1 - var1)

    def B43(var1, var2, var3, var4):
        return var4

    def B44(var1, var2, var3, var4):
        return var3

    # Cmat
    def Cmat1(var1, var2, var3, var4):
        return 0

    def Cmat2(var1, var2, var3, var4):
        return (np.sin(beta)*g*rho_l*var1 + tau_gl(var1, var2, var3, var4)*P_gl(var1)/A - tau_lw(var1, var2)*P_lw(var1)/A + Fbody*var1)/(rho_l*var1)

    def Cmat3(var1, var2, var3, var4):
        return -np.sin(beta)*g + (tau_gl(var1, var2, var3, var4)*P_gl(var1)/A + tau_gw(var1, var3, var4)*P_gw(var1)/A + Fbody*(-1 + var1))/(rho_g(var4)*(-1 + var1))

    def Cmat4(var1, var2, var3, var4):
        return 0

    # # Dmat
    # def Dmat (var1, var4, Gamma_l, Gamma_g, nu_l, nu_g):
    #     Dmat = [[rho_l*Gamma_l, 0, 0, 0],

    #             [-rho_g(var4)*Gamma_g, 0, 0, 0],

    #             [0, var1*nu_l, 0, 0],

    #             [0, 0, nu_g*(1 - var1), 0]]
    #     return Dmat


# System 2(FEniCS)

if system == 2:
   # Amat
    def A11_fenics(var1, var2, var3, var4):
        return 1

    def A12_fenics(var1, var2, var3, var4):
        return 0

    def A13_fenics(var1, var2, var3, var4):
        return 0

    def A14_fenics(var1, var2, var3, var4):
        return 0

    def A21_fenics(var1, var2, var3, var4):
        return 0

    def A22_fenics(var1, var2, var3, var4):
        return 1

    def A23_fenics(var1, var2, var3, var4):
        return 0

    def A24_fenics(var1, var2, var3, var4):
        return 0

    def A31_fenics(var1, var2, var3, var4):
        return 0

    def A32_fenics(var1, var2, var3, var4):
        return 0

    def A33_fenics(var1, var2, var3, var4):
        return 1

    def A34_fenics(var1, var2, var3, var4):
        return 0

    def A41_fenics(var1, var2, var3, var4):
        return 0

    def A42_fenics(var1, var2, var3, var4):
        return 0

    def A43_fenics(var1, var2, var3, var4):
        return 0

    def A44_fenics(var1, var2, var3, var4):
        return 1

    # Bmat
    def B11_fenics(var1, var2, var3, var4):
        return var2

    def B12_fenics(var1, var2, var3, var4):
        return var1

    def B13_fenics(var1, var2, var3, var4):
        return Constant(0.0)

    def B14_fenics(var1, var2, var3, var4):
        return Constant(0.0)

    def B21_fenics(var1, var2, var3, var4):
        return Dc_fenics(var1)

    def B22_fenics(var1, var2, var3, var4):
        return var2

    def B23_fenics(var1, var2, var3, var4):
        return Constant(0.0)

    def B24_fenics(var1, var2, var3, var4):
        return 1/rho_l

    def B31_fenics(var1, var2, var3, var4):
        return Dc_fenics(var1)

    def B32_fenics(var1, var2, var3, var4):
        return Constant(0.0)

    def B33_fenics(var1, var2, var3, var4):
        return var3

    def B34_fenics(var1, var2, var3, var4):
        return 1/rho_g(var4)

    def B41_fenics(var1, var2, var3, var4):
        return (-var2 + var3)*var4/(-1 + var1)

    def B42_fenics(var1, var2, var3, var4):
        return var1*var4/(1 - var1)

    def B43_fenics(var1, var2, var3, var4):
        return var4

    def B44_fenics(var1, var2, var3, var4):
        return var3

    # Cmat
    def Cmat1_fenics(var1, var2, var3, var4):
        return Constant(0.0)

    def Cmat2_fenics(var1, var2, var3, var4):
        return (sin(beta)*g*rho_l*var1 +
                tau_gl_fenics(var1, var2, var3, var4)*P_gl_fenics(var1)/A -
                tau_lw(var1, var2)*P_lw(var1)/A)/(rho_l*var1)

    def Cmat3_fenics(var1, var2, var3, var4):
        return -sin(beta)*g +\
            (tau_gl_fenics(var1, var2, var3, var4)*P_gl_fenics(var1)/A +
             tau_gw_fenics(var1, var3, var4)*P_gw(var1)/A)/(rho_g(var4)*(-1 + var1))

    def Cmat4_fenics(var1, var2, var3, var4):
        return Constant(0.0)

    # # Dmat
    # def Dmat(var1, var4, Gamma_l, Gamma_g, nu_l, nu_g):
    #     Dmat = [[rho_l*Gamma_l, 0, 0, 0],

    #             [-rho_g(var4)*Gamma_g, 0, 0, 0],

    #             [0, var1*nu_l, 0, 0],

    #             [0, 0, nu_g*(1 - var1), 0]]
    #     return Dmat

# U =(alpha_l*rho_l, alpha_g*rho_g, alpha_l*rho_l*u_l, alpha_g*rho_g*u_g)
# A = I

if system == 3:
    # Amat
    def A11(var1, var2, var3, var4):
        return 1

    def A12(var1, var2, var3, var4):
        return 0

    def A13(var1, var2, var3, var4):
        return 0

    def A14(var1, var2, var3, var4):
        return 0

    def A21(var1, var2, var3, var4):
        return 0

    def A22(var1, var2, var3, var4):
        return 1

    def A23(var1, var2, var3, var4):
        return 0

    def A24(var1, var2, var3, var4):
        return 0

    def A31(var1, var2, var3, var4):
        return 0

    def A32(var1, var2, var3, var4):
        return 0

    def A33(var1, var2, var3, var4):
        return 1

    def A34(var1, var2, var3, var4):
        return 0

    def A41(var1, var2, var3, var4):
        return 0

    def A42(var1, var2, var3, var4):
        return 0

    def A43(var1, var2, var3, var4):
        return 0

    def A44(var1, var2, var3, var4):
        return 1

    # Bmat
    def B11(var1, var2, var3, var4):
        return 0

    def B12(var1, var2, var3, var4):
        return 0

    def B13(var1, var2, var3, var4):
        return 1

    def B14(var1, var2, var3, var4):
        return 0

    def B21(var1, var2, var3, var4):
        return 0

    def B22(var1, var2, var3, var4):
        return 0

    def B23(var1, var2, var3, var4):
        return 0

    def B24(var1, var2, var3, var4):
        return 1

    def B31(var1, var2, var3, var4):
        return Dc(var1/rho_l)/rho_l + (var1*var2)/(p_factor*rho_l*(rho_l - var1)*(1 - var1/rho_l)) - var3**2/var1**2

    def B32(var1, var2, var3, var4):
        return var1/(rho_l*(p_factor - (p_factor*var1)/rho_l))

    def B33(var1, var2, var3, var4):
        return 2*var3/var1

    def B34(var1, var2, var3, var4):
        return 0

    def B41(var1, var2, var3, var4):
        return ((var2)/(p_factor*(1 - var1/rho_l)) + (Dc(var1/rho_l)*(p_factor - (p_factor*var1)/rho_l)*var2)/(p_factor*(1 - var1/rho_l)))/rho_l

    def B42(var1, var2, var3, var4):
        return 1/p_factor - var4**2/var2**2

    def B43(var1, var2, var3, var4):
        return 0

    def B44(var1, var2, var3, var4):
        return 2*var4/var2

    # Cmat
    def Cmat1(var1, var2, var3, var4):
        return 0

    def Cmat2(var1, var2, var3, var4):
        return 0

    def Cmat3(var1, var2, var3, var4):
        return g*np.sin(beta)*var1 + tau_lw(var1, var2)*P_lw(var1)/A + tau_gl(var1, var2, var3, var4)*P_gl(var1)/A

    def Cmat4(var1, var2, var3, var4):
        return -(np.sin(beta)*g*(var1/rho_l - 1)*var2)/(1 - var1/rho_l) + tau_gw(var1, var3, var4)*P_gw(var1)/A - tau_gl(var1, var2, var3, var4)*P_gl(var1)/A

    # # # Dmat
    # # def Dmat (var1, var4, Gamma_l, Gamma_g, nu_l, nu_g):
    # #     Dmat = [[rho_l*Gamma_l, 0, 0, 0],

    # #             [-rho_g (var4)*Gamma_g, 0, 0, 0],

    # #             [0, var1*nu_l, 0, 0],

    # #             [0, 0, nu_g*(1 - var1), 0]]
    # #     return Dmat

# Construction of the matrices
if system == 3:
    Aeval, Beval, Ceval = linear_matrices_function_conserved(
        ref[0], ref[1], ref[2], ref[3], beta, D, system, rho_l, p_factor, mu_l, mu_g, dirichlet_type)
else:
    pass


# 3. Linearized matrices
if any([simulation == 1, simulation == 2, simulation == 3, simulation == 4]):
    # Amat
    def Amat(var1, var2, var3, var4):
        Amat = [[A11(var1, var2, var3, var4), A12(var1, var2, var3, var4), A13(var1, var2, var3, var4), A14(var1, var2, var3, var4)],

                [A21(var1, var2, var3, var4), A22(var1, var2, var3, var4),
                 A23(var1, var2, var3, var4), A24(var1, var2, var3, var4)],

                [A31(var1, var2, var3, var4), A32(var1, var2, var3, var4),
                 A33(var1, var2, var3, var4), A34(var1, var2, var3, var4)],

                [A41(var1, var2, var3, var4), A42(var1, var2, var3, var4), A43(var1, var2, var3, var4), A44(var1, var2, var3, var4)]]
        return Amat

    # Bmat
    def Bmat(var1, var2, var3, var4):
        Bmat = [[B11(var1, var2, var3, var4), B12(var1, var2, var3, var4), B13(var1, var2, var3, var4), B14(var1, var2, var3, var4)],

                [B21(var1, var2, var3, var4), B22(var1, var2, var3, var4),
                 B23(var1, var2, var3, var4), B24(var1, var2, var3, var4)],

                [B31(var1, var2, var3, var4), B32(var1, var2, var3, var4),
                 B33(var1, var2, var3, var4), B34(var1, var2, var3, var4)],

                [B41(var1, var2, var3, var4), B42(var1, var2, var3, var4), B43(var1, var2, var3, var4), B44(var1, var2, var3, var4)]]
        return Bmat

    # Cmat linearization
    def Cmat1_var1(var1):
        var2 = ref[1]
        var3 = ref[2]
        var4 = ref[3]
        return Cmat1(var1, var2, var3, var4)

    def Cmat1_var2(var2):
        var1 = ref[0]
        var3 = ref[2]
        var4 = ref[3]
        return Cmat1(var1, var2, var3, var4)

    def Cmat1_var3(var3):
        var1 = ref[0]
        var2 = ref[1]
        var4 = ref[3]
        return Cmat1(var1, var2, var3, var4)

    def Cmat1_var4(var4):
        var1 = ref[0]
        var2 = ref[1]
        var3 = ref[2]
        return Cmat1(var1, var2, var3, var4)

    def Cmat2_var1(var1):
        var2 = ref[1]
        var3 = ref[2]
        var4 = ref[3]
        return Cmat2(var1, var2, var3, var4)

    def Cmat2_var2(var2):
        var1 = ref[0]
        var3 = ref[2]
        var4 = ref[3]
        return Cmat2(var1, var2, var3, var4)

    def Cmat2_var3(var3):
        var1 = ref[0]
        var2 = ref[1]
        var4 = ref[3]
        return Cmat2(var1, var2, var3, var4)

    def Cmat2_var4(var4):
        var1 = ref[0]
        var2 = ref[1]
        var3 = ref[2]
        return Cmat2(var1, var2, var3, var4)

    def Cmat3_var1(var1):
        var2 = ref[1]
        var3 = ref[2]
        var4 = ref[3]
        return Cmat3(var1, var2, var3, var4)

    def Cmat3_var2(var2):
        var1 = ref[0]
        var3 = ref[2]
        var4 = ref[3]
        return Cmat3(var1, var2, var3, var4)

    def Cmat3_var3(var3):
        var1 = ref[0]
        var2 = ref[1]
        var4 = ref[3]
        return Cmat3(var1, var2, var3, var4)

    def Cmat3_var4(var4):
        var1 = ref[0]
        var2 = ref[1]
        var3 = ref[2]
        return Cmat3(var1, var2, var3, var4)

    def Cmat4_var1(var1):
        var2 = ref[1]
        var3 = ref[2]
        var4 = ref[3]
        return Cmat4(var1, var2, var3, var4)

    def Cmat4_var2(var2):
        var1 = ref[0]
        var3 = ref[2]
        var4 = ref[3]
        return Cmat4(var1, var2, var3, var4)

    def Cmat4_var3(var3):
        var1 = ref[0]
        var2 = ref[1]
        var4 = ref[3]
        return Cmat4(var1, var2, var3, var4)

    def Cmat4_var4(var4):
        var1 = ref[0]
        var2 = ref[1]
        var3 = ref[2]
        return Cmat4(var1, var2, var3, var4)

    # Cmat_lin
    def Cmat_lin(var1, var2, var3, var4):
        Cmat_lin = [[gradient(Cmat1_var1, 1), gradient(Cmat1_var2, 2), gradient(Cmat1_var3, 3), gradient(Cmat1_var4, 4)],
                    [gradient(Cmat2_var1, 1), gradient(Cmat2_var2, 2),
                     gradient(Cmat2_var3, 3), gradient(Cmat2_var4, 4)],
                    [gradient(Cmat3_var1, 1), gradient(Cmat3_var2, 2),
                     gradient(Cmat3_var3, 3), gradient(Cmat3_var4, 4)],
                    [gradient(Cmat4_var1, 1), gradient(Cmat4_var2, 2), gradient(Cmat4_var3, 3), gradient(Cmat4_var4, 4)]]
        return Cmat_lin

# ## Functions of coefficient matrices
# Primitive variables

# FUNCTION for primitive variables:


def linear_matrices_function(var1, var2, var3, var4, beta, D, system, rho_l, p_factor, mu_l, mu_g, dirichlet_type):
    # Matrices
    Aeval = np.asarray(Amat(var1, var2, var3, var4))
    Beval = np.asarray(Bmat(var1, var2, var3, var4))
    Ceval = np.asarray(Cmat_lin(var1, var2, var3, var4))

    return Aeval, Beval, Ceval

# Conserved variables
# FUNCTION:


def linear_matrices_function_conserved(var1, var2, var3, var4, beta, D, system, rho_l, p_factor, mu_l, mu_g, dirichlet_type):

    # Matrices
    Aeval = np.asarray(Amat(var1, var2, var3, var4))
    Beval = np.asarray(Bmat(var1, var2, var3, var4))
    Ceval = np.asarray(Cmat_lin(var1, var2, var3, var4))

    return Aeval, Beval, Ceval


# (base) root@MacBook twofluidmodel # conda activate fenicsproject
# (fenicsproject) root@MacBook twofluidmodel # ./modules/equations_matrices.py
