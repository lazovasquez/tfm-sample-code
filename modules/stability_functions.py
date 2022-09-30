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
import numpy as np


# Characteristics
def characteristics_function(Aeval, Beval):
    m1, _ = eig(Beval, b=Aeval, overwrite_a=True,
                overwrite_b=True, check_finite=True)

    listreal1 = []
    listimag1 = []

    for l1 in range(len(m1)):
        realpart1 = m1[l1].real
        imagpart1 = m1[l1].imag

        listimag1.append(imagpart1)
        listreal1.append(realpart1)

    maxchar = max(listreal1)

    return listreal1, listimag1, maxchar


# # Amplification factors and eigenvectors
# > Matrices A,B,C
def stability_function(Aeval, Beval, Ceval):
    Acomplex = Aeval.dot(1j)
    Bcomplex = Beval.dot(1j)

    m2, vect2 = eig((-wavenumber_fourier*Bcomplex - Ceval), b=-Acomplex,
                    overwrite_a=True, overwrite_b=True, check_finite=True)

    listreal2 = []
    listimag2 = []

    for l2 in range(len(m2)):
        realpart2 = m2[l2].real
        imagpart2 = m2[l2].imag

        listimag2.append(imagpart2)
        listreal2.append(realpart2)

    return listreal2, listimag2, m2, vect2


# > Matrix A
def stability_function_A(Aeval):
    m2, _ = eig(Aeval, check_finite=True)

    listreal2 = []
    listimag2 = []

    for l2 in range(len(m2)):
        realpart2 = m2[l2].real
        imagpart2 = m2[l2].imag

        listimag2.append(imagpart2)
        listreal2.append(realpart2)

    return listreal2, listimag2


# Stiffness Solver
def solver_stiffness(matrix):
    eigensolver = SLEPcEigenSolver(matrix)  # (A, B)
    # PETScOptions.set ("eps_view")
    # OPTIONS: “power” (power iteration)
    eigensolver.parameters["solver"] = "subspace"
    # “subspace” (subspace iteration)
    # “arnoldi” (Arnoldi) “lanczos” (Lanczos)
    # “krylov-schur” (Krylov-Schur)
    # “lapack” (LAPACK, all values, direct, small systems only)
    # “arpack” (ARPACK)
    # OPTIONS: “hermitian” (Hermitian)
    eigensolver.parameters["problem_type"] = "non_hermitian"
    # “non_hermitian” (Non-Hermitian)
    # “gen_hermitian” (Generalized Hermitian)
    # “gen_non_hermitian” (Generalized Non-Hermitian)
    # “pos_gen_non_hermitian” (Generalized Non-Hermitian with positive semidefinite B)
    # eigensolver.parameters["maximum_iterations"] = 1000 # OPTIONS:
    # eigensolver.parameters["tolerance"] = 1e-15 # The default is 1e-15;
    # eigensolver.parameters["spectrum"] = "target magnitude" # OPTIONS: "target magnitude", "largest magnitude"
    # eigensolver.parameters["spectral_transform"] = "shift-and-invert" # OPTIONS: "shift-and-invert"
    # eigensolver.parameters["spectral_shift"] = 0.9
    # neigs = 12

    # Compute all eigenvalues of A x = \lambda x
    eigensolver.solve()  # (neigs)

    # Exporting the real part of the eigenvectors and plotting eigenvalues
    Real = []
    Imag = []

    for i in range(eigensolver.get_number_converged()):
        r, c, _, _ = eigensolver.get_eigenpair(i)

        # Real part of eigenvalues
        Real.append(r)
        Imag.append(c)

    return Real, Imag


# Function
def stiffness_function(Bm, visc, Cm, variable, dvariable):
    # Variational form
    R = Bm + Cm

    # Compute directional derivative about u in the direction of du (Jacobian)
    dF = derivative(R, variable, dvariable)
    dummy = (inner(Constant(0.0), v1) + inner(Constant(0.0), v2) +
             inner(Constant(0.0), v3) + inner(Constant(0.0), v4))*dx  # Alternative 2

    # Assemble stiffness form
    A_stiffness = PETScMatrix()  # Alternative 1, # Alternative 2
    b_stiffness = PETScVector()  # Alternative 2

    # Assemble system
    # bcs = [] # Alternative 2
    # A_ufl, _ = assemble_system (dF, dummy, bcs = bcs, A_tensor = A_stiffness, b_tensor = b_stiffness) # Alternative 2
    A_ufl = assemble(dF, tensor=A_stiffness)  # Alternative 1

    # Matrix A1
    A_array = matrix(A_ufl.array())

    # Condition number
    condnumber = LA.cond(A_array)
    print("Condition number:", condnumber)

    Real, Imag = solver_stiffness(A_ufl)
    # Real, Imag = stability_function_A (A_array)

    return Real, Imag, A_ufl, A_array


# > Eigenvalue function of a sparse matrix
def eigenvalue_function(Beval):
    m1, _ = eigs(Beval)

    listreal1 = []
    listimag1 = []

    for l1 in range(len(m1)):
        realpart1 = m1[l1].real
        imagpart1 = m1[l1].imag

        listimag1.append(imagpart1)
        listreal1.append(realpart1)

    maxchar = max(listreal1)

    return listreal1, listimag1, maxchar

# > Eigenvalues of the stiffness matrix

# listreal1, listimag1, maxchar = eigenvalue_function (A_sparray)

# #################### Plot convective waves
# plt.figure ()
# fig, ax = plt.subplots ()
# ax.scatter (listreal1,
#             listimag1,
#             s = area_scatter,
#             marker = listmarkers [0],
#             color = listcolor [4],
#             edgecolors = listcolor [0],
#             linewidths = line_width,
#             alpha = alphascatter)

# # plt.rcParams ['figure.figsize'] = mapsize
# # leg1 = ax.legend (loc = 'upper right', frameon = True, fontsize = 14);
# plt.grid (True, which = "both")
# plt.xlabel (r'Re ($\mu$) $[\it{s^{-1}}]$', fontsize = label_size)
# plt.ylabel (r'Im ($\mu$) $[\it{s^{-1}}]$', fontsize = label_size)
# # plt.xlim (-0.08, 0.02)
# # plt.ylim (-30, 30)
# matplotlib.rc ('xtick', labelsize = label_size)
# matplotlib.rc ('ytick', labelsize = label_size)

# plt.axvline (0, label = 'pyplot vertical line', color = 'k')

# # Save figure
# # fig.set_size_inches (mapsize)
# # plt.savefig('figures/semi_disc/fig3.pdf',
# #             optimize = True,
# #             transparent = True,
# #             dpi = dpi_elsevier)

# # Show figure
# plt.show

# print(maxchar)

# (base) root@MacBook twofluidmodel # conda activate fenicsproject
# (fenicsproject) root@MacBook twofluidmodel # ./modules/variational_form.py
