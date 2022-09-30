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

# Boundary conditions
if any([elementspace == 1, elementspace == 2, elementspace == 3]):
    # Inflow Dirichlet boundary condition
    def Inflow(x, on_boundary):
        return (x[0] < DOLFIN_EPS) and on_boundary
    # Ouflow Dirichlet boundary condition

    def Outflow(x, on_boundary):
        return (x[0] > (L - DOLFIN_EPS)) and on_boundary
if elementspace == 4:
    parameters["ghost_mode"] = "shared_facet"
    # https://fenicsproject.org/docs/dolfin/2016.2.0/python/demo/documented/subdomains-poisson/python/documentation.html
    # Inflow Dirichlet boundary condition

    class Inflow (SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] < DOLFIN_EPS) and on_boundary  # near
    # Ouflow Dirichlet boundary condition

    class Outflow (SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] > (L - DOLFIN_EPS)) and on_boundary  # near
    # Initialize sub-domain instances
    inflow = Inflow()
    outflow = Outflow()

# Periodic boundary conditions


class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > - DOLFIN_EPS and on_boundary)
        # return bool (x[0] < (l + DOLFIN_EPS) and x[0] > (l - DOLFIN_EPS) and on_boundary)
    # Map right boundary (H) to left boundary (G)

    def map(self, x, y):
        y[0] = x[0] - l
        #   y[0] = l - x[0]


# Define Dirichlet boundary conditions
if any([system == 1, system == 2]):
    # Define Dirichlet boundary conditions for smooth flow
    if IBVP == 1:
        variable1_dirichlet = Constant(ref[0])
        variable2_dirichlet = Constant(ref[1])
        variable3_dirichlet = Constant(ref[2])
        variable4_dirichlet = Constant(ref[3])

    # Define Dirichlet boundary conditions for perturbed flow
    elif IBVP == 2:
        # Initial time
        tbc = 0

        # Perturbation wave
        variable1_dirichlet = Expression(
            'reference1 + amplitude1*sin (frequency1*tbc - wavenumber*x[0])',
            degree=deg1,
            reference1=Constant(ref[0]),
            amplitude1=(vect2[0]).real,
            wavenumber=wavenumber_fourier,
            frequency1=listreal2[0],
            tbc=tbc,
            domain=mesh)
        variable2_dirichlet = Expression(
            'reference2 + amplitude2*sin (frequency2*tbc - wavenumber*x[0])',
            degree=deg2,
            reference2=Constant(ref[1]),
            amplitude2=(vect2[1]).real,
            wavenumber=wavenumber_fourier,
            frequency2=listreal2[1],
            tbc=tbc,
            domain=mesh)
        variable3_dirichlet = Expression(
            'reference3 + amplitude3*sin (frequency3*tbc - wavenumber*x[0])',
            degree=deg3,
            reference3=Constant(ref[2]),
            amplitude3=(vect2[2]).real,
            wavenumber=wavenumber_fourier,
            frequency3=listreal2[2],
            tbc=tbc,
             domain=mesh)
        variable4_dirichlet = Expression(
            'reference4 + amplitude4*sin (frequency4*tbc - wavenumber*x[0])',
            degree=deg4,
            reference4=Constant(ref[3]),
            amplitude4=(vect2[3]).real,
            wavenumber=wavenumber_fourier,
            frequency4=listreal2[3],
            tbc=tbc,
            domain=mesh)
elif system == 3:
    if IBVP == 1:
        # Define Dirichlet boundary conditions for smooth flow
        variable1_dirichlet = Constant(ref[0]/rho_l)
        variable2_dirichlet = Constant(ref[2]/ref[0])
        variable3_dirichlet = Constant(ref[3]/ref[1])
        variable4_dirichlet = Constant(ref[1]/((1 - ref[0]/rho_l)*p_factor))


# Define vector of boundary conditions
if any([elementspace == 1, elementspace == 2, elementspace == 3]):
    # Dirichlet boundary conditions
    # , "geometric") # "geometric" "pointwise"
    bc1 = DirichletBC(V.sub(0), variable1_dirichlet, Inflow)
    bc2 = DirichletBC(V.sub(1), variable2_dirichlet, Inflow)  # , "geometric")
    bc3 = DirichletBC(V.sub(2), variable3_dirichlet, Inflow)  # , "geometric")
    bc4 = DirichletBC(V.sub(3), variable4_dirichlet, Outflow)  # , "geometric")
    bcs = [bc1, bc2, bc3, bc4]

elif elementspace == 4:
    # Define ds for facets
    # DOLFIN predefines the “measures” dx, ds and dS representing integration over cells, exterior facets (that is, facets on the boundary) and interior facets, respectively. These measures can take an additional integer argument. In fact, dx defaults to dx(0), ds defaults to ds(0), and dS defaults to dS(0). Integration over subregions can be specified by measures with different integer labels as arguments.
    # Define outer surface measure aware of Dirichlet boundaries
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    # Dirichlet boundary conditions
    # , "geometric") # "geometric"<<<<
    bc1 = DirichletBC(V.sub(0), variable1_dirichlet, boundaries, 1)
    bc2 = DirichletBC(V.sub(1), variable2_dirichlet,
                      boundaries, 1)  # , "geometric")
    bc3 = DirichletBC(V.sub(2), variable3_dirichlet,
                      boundaries, 1)  # , "geometric")
    bc4 = DirichletBC(V.sub(3), variable4_dirichlet,
                      boundaries, 2)  # , "geometric")
    bcs = [bc1, bc2, bc3, bc4]


# Initial conditions
if IBVP == 1:
    class InitialConditions_SS (UserExpression):
        def eval(self, values, x):
            values[0] = ref[0]
            values[1] = ref[1]
            values[2] = ref[2]
            values[3] = ref[3]

        def value_shape(self):
            return (4,)
elif IBVP == 2:
    class InitialConditions_SS_perturbed (UserExpression):
        def eval(self, values, x):
            amplitude1 = (eigenvector[0]).real
            amplitude2 = (eigenvector[1]).real
            amplitude3 = (eigenvector[2]).real
            amplitude4 = (eigenvector[3]).real

            values[0] = ref[0] + amplitude1*cos(-wavenumber_fourier*x[0])
            values[1] = ref[1] + amplitude2*cos(-wavenumber_fourier*x[0])
            values[2] = ref[2] + amplitude3*cos(-wavenumber_fourier*x[0])
            values[3] = ref[3] + amplitude4*cos(-wavenumber_fourier*x[0])

        def value_shape(self):
            return (4,)

# Initial conditions
if IBVP == 1:
    variable_init = InitialConditions_SS()
elif IBVP == 2:
    variable_init = InitialConditions_SS_perturbed()

# interpolate vector of initial conditions
variable_ic.interpolate(variable_init)

# interpolate vector of initial conditions (transient simulations)
if any([time_method == 1, time_method == 3]):
    variable_n.interpolate(variable_init)
elif time_method == 2:
    variable_past.interpolate(variable_init)


# PLOTS
# Show initial conditions
if show_data == 1:
    if any([system == 1, system == 2]):
        plt.figure(1)
        fig, ax = plt.subplots()
        # plt.ylim (0, 1)
        plt.xlim(0, L)
        plt.grid(True, which="both")
        ax.set_xlabel(r'L [m]')
        ax.set_ylabel(r'$\alpha_l$ [-]')
        ax.xaxis.set_tick_params(which='major', direction='in', top='on')
        ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
        ax.yaxis.set_tick_params(which='major', direction='in', right='on')
        ax.yaxis.set_tick_params(which='minor', direction='in', right='on')
        plot(variable1_ic, color='k')

        # Save plot
        fig.set_size_inches(mapsize)
        plt.savefig('results/figures/initial_conditions/fig1.pdf',
                    optimize=True,
                    transparent=True,
                    dpi=dpi_elsevier)

        plt.figure(2)
        fig, ax = plt.subplots()
        plt.xlim(0, L)
        plt.grid(True, which="both")
        ax.set_xlabel(r'L [m]')
        ax.set_ylabel(r'$u_l$ [m/s]')
        ax.xaxis.set_tick_params(which='major', direction='in', top='on')
        ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
        ax.yaxis.set_tick_params(which='major', direction='in', right='on')
        ax.yaxis.set_tick_params(which='minor', direction='in', right='on')
        plot(variable2_ic, color='k')

        # Save plot
        fig.set_size_inches(mapsize)
        plt.savefig('results/figures/initial_conditions/fig2.pdf',
                    optimize=True,
                    transparent=True,
                    dpi=dpi_elsevier)

        plt.figure(3)
        fig, ax = plt.subplots()
        plt.xlim(0, L)
        plt.grid(True, which="both")
        ax.set_xlabel(r'L [m]')
        ax.set_ylabel(r'$u_g$ [m/s]')
        ax.xaxis.set_tick_params(which='major', direction='in', top='on')
        ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
        ax.yaxis.set_tick_params(which='major', direction='in', right='on')
        ax.yaxis.set_tick_params(which='minor', direction='in', right='on')
        plot(variable3_ic, color='k')

        # Save plot
        fig.set_size_inches(mapsize)
        plt.savefig('results/figures/initial_conditions/fig3.pdf',
                    optimize=True,
                    transparent=True,
                    dpi=dpi_elsevier)

        plt.figure(4)
        fig, ax = plt.subplots()
        plt.xlim(0, L)
        plt.grid(True, which="both")
        ax.set_xlabel(r'L [m]')
        ax.set_ylabel(r'$p_i$ [Pa]')
        ax.xaxis.set_tick_params(which='major', direction='in', top='on')
        ax.xaxis.set_tick_params(which='minor', direction='in', top='on')
        ax.yaxis.set_tick_params(which='major', direction='in', right='on')
        ax.yaxis.set_tick_params(which='minor', direction='in', right='on')
        plot(variable4_ic, color='k')

        # Save plot
        fig.set_size_inches(mapsize)
        plt.savefig('results/figures/initial_conditions/fig4.pdf',
                    optimize=True,
                    transparent=True,
                    dpi=dpi_elsevier)

# (base) root@MacBook twofluidmodel # conda activate fenicsproject
# (fenicsproject) root@MacBook twofluidmodel # ./modules/initial_boundary_conditions.py
