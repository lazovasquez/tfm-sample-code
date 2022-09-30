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


# # Test points beta
# testpointsbeta = len(betavector)

# # Scatter of test cases in maps
# testpointsj_l_scatter = len(j_lscatter)
# testpointsj_g_scatter = len(j_gscatter)


# GEOMETRY
# Cross-sectional area
A = pi*pow(D, 2.)/4


# liquid sectional area
def A_l(var1):
    return A*var1


# gas sectional area
def A_g(var1):
    return A*(1 - var1)


# stratification angle
def gamma(var1):
    # return pi*(var1) + pow((3.*pi/2.), (1./3.))*(1 - 2*(var1) + pow(abs(var1), (1./3.)) - pow(abs(1 - (var1)), (1/3))) - (1/200)*(1 - (var1))*(var1)*(1 - 2*(var1))*(1+4*((1 - (var1))**2 + (var1)**2))
    # return pi*var1 + pow((3.*pi/2.), (1./3.))*(1 - 2*var1 + pow(abs(var1), (1./3.)) - pow(abs(1 - var1), (1/3))) - (1/200)*(1-var1)*var1*(1 - 2*var1)*(1+4*((1-var1)**2+(var1)**2))
    return pi*var1 + ((3*pi/2)**(1/3))*(1 - 2*var1 + (abs(var1 + DOLFIN_EPS)**(1/3)) - (abs(1 - (var1))**(1/3))) - (1/200)*(1 - var1)*var1*(1 - 2*var1)*(1 + 4*((1-var1)**2 + (var1)**2))


def gamma_fenics(var1):
    return pi*var1 + ((3*pi/2)**(1/3))*(1 - 2*var1 + (abs(var1 + DOLFIN_EPS)**(1/3)) - (abs(1 - (var1 + DOLFIN_EPS))**(1/3))) - (1/200)*(1 - var1)*var1*(1 - 2*var1)*(1 + 4*((1-var1)**2 + (var1)**2))


# liquid wetted perimeter
def P_lw(var1):
    return D*gamma(var1)


# gas wetted perimeter
def P_gw(var1):
    return D*(pi - gamma(var1))


# interface wetted perimeter
def P_gl(var1):
    return D*np.sin(gamma(var1))


# interface wetted perimeter
def P_gl_fenics(var1):
    return D*sin(gamma(var1))


# critic diameter
def Dc(var1):
    return g*np.cos(beta)*pi*D/(4*np.sin(gamma(var1)))


def Dc_fenics(var1):
    return g*cos(beta)*pi*D/(4*sin(gamma(var1)))


# liquid hydraulic diameter
def Dh_l(var1):
    return 4.*A_l(var1)/P_lw(var1)


# gas hydraulic diameter
def Dh_g(var1):
    return 4.*A_g(var1)/(P_gw(var1) + P_gl(var1))


def Dh_g_fenics(var1):
    return 4.*A_g(var1)/(P_gw(var1) + P_gl_fenics(var1))


# FLUID PROPERTIES
# Gas density
def rho_g(var4):
    return p_factor*var4 + DOLFIN_EPS


# Compressibility factor
p_factor = 1/c_g**2


# Flow properties
# liquid phase velocity
def var2(var1):
    return j_l/(var1)


# gas phase velocity
def var3(var1):
    return j_g/(1 - var1)


# FRICTION
# liquid Reynolds number
def Re_l(var1, var2):
    return rho_l*var2*Dh_l(var1)/mu_l


# gas Reynolds number
def Re_g(var1, var3, var4):
    return rho_g(var4)*var3*Dh_g(var1)/mu_g


def Re_g_fenics(var1, var3, var4):
    return rho_g(var4)*var3*Dh_g_fenics(var1)/mu_g


# Friction factor liquid-wall
def f_lw(var1, var2):
    return 0.046*abs(Re_l(var1, var2) + DOLFIN_EPS)**(-0.2)

# # Friction factor liquid-wall (laminar)
# def f_lw_laminar(var1, var2):
#     return 24/Re_l(var1, var2)
# def f_lw_laminar_fenics(var1, var2):
#     return 24/Re_l(var1, var2)


# Friction factor gas-wall
def f_gw(var1, var3, var4):
    return 0.046*abs(Re_g(var1, var3, var4) + DOLFIN_EPS)**(-0.2)


def f_gw_fenics(var1, var3, var4):
    return 0.046*abs(Re_g_fenics(var1, var3, var4) + DOLFIN_EPS)**(-0.2)

# # Friction factor gas-wall (laminar)
# def f_gw_laminar(var1, var3, var4):
#     return 16 / Re_g(var1, var3, var4)
# def f_gw_laminar_fenics(var1, var3, var4):
#     return 16 / Re_g_fenics(var1, var3, var4)


# interfacial friction
def f_gl(var1, var2, var3, var4):
    return max(f_gw(var1, var3, var4), 0.014)


def f_gl_fenics(var1, var2, var3, var4):
    return Max(f_gw_fenics(var1, var3, var4), 0.014)


# Shear stress liquid-wall
def tau_lw(var1, var2):
    return 1/2*f_lw(var1, var2)*rho_l*var2*var2


# gas shear stress
def tau_gw(var1, var3, var4):
    return 1/2*f_gw(var1, var3, var4)*rho_g(var4)*var3*abs(var3)


def tau_gw_fenics(var1, var3, var4):
    return 1/2*f_gw_fenics(var1, var3, var4)*rho_g(var4)*var3*abs(var3)


# interface shear stress l
def tau_gl(var1, var2, var3, var4):
    return 1/2*f_gl(var1, var2, var3, var4)*rho_g(var4)*(var3 - var2)*abs(var3 - var2)


def tau_gl_fenics(var1, var2, var3, var4):
    return 1/2*f_gl_fenics(var1, var2, var3, var4)*rho_g(var4)*(var3 - var2)*abs(var3 - var2)


# STRATIFIED FLOW
# Equilibrium
def equilibrium1(var1):
    return (rho_g(var4_0) - rho_l)*g*np.sin(beta) - tau_lw(var1, var2(var1))*P_lw(var1)/A_l(var1) + tau_gw(var1, var3(var1), var4_0)*P_gw(var1)/A_g(var1) + tau_gl(var1, var2(var1), var3(var1), var4_0)*P_gl(var1)*(1/A_l(var1) + 1/A_g(var1))


# FUNCTIONS FOR REFERENCE USING J_L
# liquid phase velocity
def var2b(j_l):
    return j_l/(var1)


# Equilibrium
def equilibrium2(j_l):
    return (rho_g(var4_0) - rho_l)*g*np.sin(beta) - tau_lw(var1, var2b(j_l))*P_lw(var1)/A_l(var1) + tau_gw(var1, var3(var1), var4_0)*P_gw(var1)/A_g(var1) + tau_gl(var1, var2b(j_l), var3(var1), var4_0)*P_gl(var1)*(1/A_l(var1) + 1/A_g(var1))


# 2. Reference state functions
# Find var1 with given j_l and j_g
def ref_state(j_l, j_g, var4_0, beta, rho_l, p_factor, mu_l, mu_g, D):
    # Reference state
    var1_ref = brenth(equilibrium1, lima, limb)
    var2_ref = j_l/var1_ref
    var3_ref = j_g/(1 - var1_ref)
    var4_ref = var4_0

    # Reference conditions
    ref = np.array([var1_ref, var2_ref, var3_ref, var4_ref])

    # Reynolds number of initial conditions
    Rel_ref = Re_l(ref[0], ref[1])
    Reg_ref = Re_g(ref[0], ref[2], ref[3])

    # Reynolds reference conditions
    Re_ref = np.array([Rel_ref, Reg_ref])

    return ref, Re_ref


# Find j_l with given alpha_l
def ref_state_jl(var1, j_g, var4_0, beta, rho_l, p_factor, mu_l, mu_g, D):
    # Define j_l
    j_l = fsolve(equilibrium2, x0)

    # Reference state
    var1_ref = var1
    var2_ref = j_l[0]/var1_ref
    var3_ref = j_g/(1 - var1_ref)
    var4_ref = var4_0

    # Reference conditions
    ref = np.array([var1_ref, var2_ref, var3_ref, var4_ref])
    return ref

# (base) root@MacBook twofluidmodel # conda activate fenicsproject
# (fenicsproject) root@MacBook twofluidmodel # ./modules/constants_codes.py
