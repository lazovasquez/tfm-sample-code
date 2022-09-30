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

if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Transient simulator for two-phase flows')
    parser.add_argument('--case',
                        action='store',
                        metavar='case',
                        type=int,
                        help='Set of parameters.')
    arg = parser.parse_args()

    # Print Centerlines Modification Module
    print('')
    print('TWOFLUIDMODEL::SAYS:')

    # Load case parameters from case_x.json
    with open(os.path.join('cases', f'case_{arg.case}.json'), mode='r') as file1:
        case_data = json.load(file1)  # encoding='utf-8'

    # SETUP
    simulation = case_data['setup']['simulation']

    # EQUATIONS
    # Equation system
    system = case_data['setup']['equations']['system']
    viscous_terms = case_data['setup']['equations']['viscous_terms']
    # Boundary conditions
    dirichlet_type = case_data['setup']['equations']['dirichlet_type']
    # Initial conditions
    IBVP = case_data['setup']['equations']['IBVP']
    effect = case_data['setup']['equations']['effect']
    # Constants
    g = case_data['setup']['equations']['constants']['gravity']

    # PLASIC PROPERTIES
    # Liquid
    rho_l = case_data['phasic_properties']['liquid']['properties']['density']
    mu_l = case_data['phasic_properties']['liquid']['properties']['dynamic_viscosity']

    # Gas
    mu_g = 1.8e-5
    c_g = case_data['phasic_properties']['gas']['properties']['compressibility']
    var4_0 = case_data['phasic_properties']['interface']['outlet_pressure']

    # GEOMETRY
    # Pipe inclination
    inclination = case_data['geometry']['inclination']

    # NUMERICAL METHOD
    # Spatial discretization
    elementspace = case_data['numerical_method']['discretization']['space']['elementspace']
    p = case_data['numerical_method']['discretization']['space']['order']
    CFL = case_data['numerical_method']['discretization']['CFL']
    nx = case_data['numerical_method']['discretization']['space']['elements_number']

    # Time discretization
    time_method = case_data['numerical_method']['discretization']['time']['time_method']
    time_steps = case_data['numerical_method']['discretization']['time']['time_steps']

    # TRANSIENT SIMULATIONS
    # Stability
    transient_eigenspectrum = case_data['stability']['transient_spectrum']

    # VISUALIZATION
    # Plot reference conditions
    show_data = case_data['setup']['visualization']['show_data']

# (base) root@MacBook twofluidmodel # conda activate fenicsproject
# (fenicsproject) root@MacBook twofluidmodel # ./transient_simulator.py --case 1
