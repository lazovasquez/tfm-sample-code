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


# Fully discretized
# Loop for each time step.

if any([simulation == 3, simulation == 4]):
    if any([system == 1, system == 2]):
        # L2 norm for each time step
        L2norm_variable1 = []
        L2norm_variable2 = []
        L2norm_variable3 = []
        L2norm_variable4 = []

        # Time vector for L2 norm computation
        timevector = np.linspace(0, T, num_steps)

        # Step in time
        t = 0
        while (t < T):
            # Condition for Dirichlet boundary conditions
            if IBVP == 2:
                variable1_dirichlet.timedirichlet = t
                variable2_dirichlet.timedirichlet = t
                variable3_dirichlet.timedirichlet = t
                variable4_dirichlet.timedirichlet = t

            # Initial conditions
            if any([time_method == 1, time_method == 3]):
                if t == 0:
                    (variable1_n, variable2_n, variable3_n,
                     variable4_n) = variable_n.split(deepcopy=True)
                    # Compute  nodal values (initial)
                    n_val1_n = np.array(variable1_n.vector())
                    n_val2_n = np.array(variable2_n.vector())
                    n_val3_n = np.array(variable3_n.vector())
                    n_val4_n = np.array(variable4_n.vector())

                    nodal_variable1_n = n_val1_n[::-1]
                    nodal_variable2_n = n_val2_n[::-1]
                    nodal_variable3_n = n_val3_n[::-1]
                    nodal_variable4_n = n_val4_n[::-1]

                    # print ("n", nodal_variable1_n)

                    nodes_variable1_n = len(nodal_variable1_n)
                    nodes_variable2_n = len(nodal_variable2_n)
                    nodes_variable3_n = len(nodal_variable3_n)
                    nodes_variable4_n = len(nodal_variable4_n)

                    # print ("dof subspace variable1 =", nodes_variable1_n)
                    # print ("dof subspace variable2 =", nodes_variable2_n)
                    # print ("dof subspace variable3 =", nodes_variable3_n)
                    # print ("dof subspace variable4 =", nodes_variable4_n)

                    # Compute vertex values (initial step)
                    v_variable1_n = variable1_n.compute_vertex_values(mesh)
                    v_variable2_n = variable2_n.compute_vertex_values(mesh)
                    v_variable3_n = variable3_n.compute_vertex_values(mesh)
                    v_variable4_n = variable4_n.compute_vertex_values(mesh)

                    vert_variable1_n = []
                    vert_variable2_n = []
                    vert_variable3_n = []
                    vert_variable4_n = []

                    for cond_vert in range(len(coordinates)):
                        vert_variable1_n.append(v_variable1_n[cond_vert])
                        vert_variable2_n.append(v_variable2_n[cond_vert])
                        vert_variable3_n.append(v_variable3_n[cond_vert])
                        vert_variable4_n.append(v_variable4_n[cond_vert])

                    vert_variable1_n = np.asarray(vert_variable1_n)
                    vert_variable2_n = np.asarray(vert_variable2_n)
                    vert_variable3_n = np.asarray(vert_variable3_n)
                    vert_variable4_n = np.asarray(vert_variable4_n)

                    print("variable1_n = ", vert_variable1_n)

                    # if simulation == 'linear_simulation':

                    # Well-posedness check
                    i = 0
                    for variable1_test, variable2_test, variable3_test, variable4_test in zip(vert_variable1_n, vert_variable2_n, vert_variable3_n, vert_variable4_n):
                        Aeval = np.asarray(
                            Amat(variable1_test, variable2_test, variable3_test, variable4_test))
                        Beval = np.asarray(
                            Bmat(variable1_test, variable2_test, variable3_test, variable4_test))

                        m_test, _ = eig(
                            Beval, b=Aeval, overwrite_a=True, overwrite_b=True, check_finite=True)

                        listreal = []
                        listimag = []

                        for cond0 in range(len(m_test)):
                            realpart = m_test[cond0].real
                            imagpart = m_test[cond0].imag

                            listimag.append(imagpart)
                            listreal.append(realpart)

                        if any([cond1 != 0 for cond1 in listimag]):
                            print("INFO: ill-posed equation system")
                            break
                        else:
                            i += 1
                        break

                    # Condition for low liquid level of transition to intermittent flow
                    if any([cond2 <= 0 for cond2 in vert_variable1_n]):
                        print(
                            "INFO: gas phase dominant. Low liquid level. Liquid equations vanish")
                        break
                    elif any([cond2 >= 1 for cond2 in vert_variable1_n]):
                        print(
                            "INFO: liquid phase dominant. Intermittent flow. Gas equations vanish")
                        break

                    # Condition for positive velocities and pressure
                    if any([cond3 <= 0 for cond3 in vert_variable2_n]):
                        print("INFO: negative liquid velocity")
                        break

                    if any([cond4 <= 0 for cond4 in vert_variable3_n]):
                        print("INFO: negative gas velocity")
                        break

                    if any([cond5 <= 0 for cond5 in vert_variable4_n]):
                        print("INFO: negative pressure")
                        break

                    # # Reynolds check
                    # Rel_n = Re_l (vert_variable1_n, vert_variable2_n)
                    # Reg_n = Re_g (vert_variable1_n, vert_variable3_n, vert_variable4_n)

                    # if any ([ cond3 <= 1180 for cond3 in Rel_n]):
                    # # Rel < 1180:
                    #     print ("Liquid laminar")
                    # elif  any ([ cond3 > 1180 for cond3 in Rel_n]):
                    #     pass

                    # if any ([ cond4 <= 1180 for cond4 in Reg_n]):
                    #     print ("Gas laminar")
                    # elif  any ([ cond4 > 1180 for cond4 in Reg_n]):
                    #     pass

                    # Plot solution var1
                    plt.figure(5)
                    plt.xlim(0, L)

                    matplotlib.rc('xtick', labelsize=label_size)
                    matplotlib.rc('ytick', labelsize=label_size)

                    # plt.ylim (0, 1)
                    plt.xlabel(r'L [m]', fontsize=label_size)
                    plt.ylabel(r'$\alpha_l$ [-]', fontsize=label_size)
                    plt.grid(True, which="both")
                    # plt.legend(['Step1'])
                    plot(variable1_n, label="step1", rescale=False)

                    # Plot solution var2
                    plt.figure(6)
                    plt.xlim(0, L)

                    matplotlib.rc('xtick', labelsize=label_size)
                    matplotlib.rc('ytick', labelsize=label_size)

                    # plt.ylim (min (vert_variable2_n), max (vert_variable2_n))
                    plt.xlabel(r'L [m]', fontsize=label_size)
                    plt.ylabel(r'$u_l$ [m/s]', fontsize=label_size)
                    plt.grid(True, which="both")
                    plot(variable2_n, rescale=False)

                    # Plot solution var3
                    plt.figure(7)
                    plt.xlim(0, L)

                    matplotlib.rc('xtick', labelsize=label_size)
                    matplotlib.rc('ytick', labelsize=label_size)

                    plt.xlabel(r'L [m]', fontsize=label_size)
                    plt.ylabel(r'$u_g$ [m/s]', fontsize=label_size)
                    plt.grid(True, which="both")
                    plot(variable3_n, rescale=False)

                    # Plot solution var4
                    plt.figure(8)
                    plt.xlim(0, L)

                    matplotlib.rc('xtick', labelsize=label_size)
                    matplotlib.rc('ytick', labelsize=label_size)

                    plt.xlabel(r'L [m]', fontsize=label_size)
                    plt.ylabel(r'$p_i$ [Pa]', fontsize=label_size)
                    plt.grid(True, which="both")
                    plot(variable4_n, rescale=False)

                    # Save solution
                    ff_variable1 << variable1_n
                    ff_variable2 << variable2_n
                    ff_variable3 << variable3_n
                    ff_variable4 << variable4_n

                # Time stepping
                t += dt

                # Print progress
                print("Iteration :", int(round(t/dt)), "of", num_steps)
                print("Time      :", t, "s")

                # Amat
                Am1 = ((a0*variable1 + a1*variable1_n +
                       a2*variable1_past)/dt*v1)*dx
                Am2 = ((a0*variable2 + a1*variable2_n +
                       a2*variable1_past)/dt*v2)*dx
                Am3 = ((a0*variable3 + a1*variable3_n +
                       a2*variable1_past)/dt*v3)*dx
                Am4 = ((a0*variable4 + a1*variable4_n +
                       a2*variable1_past)/dt*v4)*dx

                Am = Am1 + Am2 + Am3 + Am4

                # Variational form
                F = visc*Cm - Bm
                F_n = visc*Cm_n - Bm_n
                R = Am - theta*F - (1 - theta)*F_n

                # Compute directional derivative about u in the direction of du (Jacobian)
                dF = derivative(R, variable, dvariable)

                # Define transient solver function
                transientsolver = solver_linear(R, variable, bcs, dF)
                transientsolver.solve()

                # Split of the vector var
                (variable1, variable2, variable3,
                 variable4) = variable.split(deepcopy=True)

                # L2 norm
                # infonorm = variable1.vector ().norm("l2")
                # print ("Solution vector norm (0): {!r}".format (infonorm))
                L2norm_variable1.append(variable1.vector().norm("l2"))
                L2norm_variable2.append(variable2.vector().norm("l2"))
                L2norm_variable3.append(variable3.vector().norm("l2"))
                L2norm_variable4.append(variable4.vector().norm("l2"))

                # Nodal values (current)
                n_val1_n1 = np.array(variable1.vector())
                n_val2_n1 = np.array(variable2.vector())
                n_val3_n1 = np.array(variable3.vector())
                n_val4_n1 = np.array(variable4.vector())

                nodal_variable1_n1 = n_val1_n1[::-1]
                nodal_variable2_n1 = n_val2_n1[::-1]
                nodal_variable3_n1 = n_val3_n1[::-1]
                nodal_variable4_n1 = n_val4_n1[::-1]

                # print ("n+1", nodal_variable1_n1)

                # Compute vertex values (next step)
                vert_variable1 = variable1.compute_vertex_values(mesh)
                vert_variable2 = variable2.compute_vertex_values(mesh)
                vert_variable3 = variable3.compute_vertex_values(mesh)
                vert_variable4 = variable4.compute_vertex_values(mesh)

                vert_variable1_n1 = []
                vert_variable2_n1 = []
                vert_variable3_n1 = []
                vert_variable4_n1 = []

                for cond_vert in range(len(coordinates)):
                    vert_variable1_n1.append(vert_variable1[cond_vert])
                    vert_variable2_n1.append(vert_variable2[cond_vert])
                    vert_variable3_n1.append(vert_variable3[cond_vert])
                    vert_variable4_n1.append(vert_variable4[cond_vert])

                vert_variable1_n1 = np.asarray(vert_variable1_n1)
                vert_variable2_n1 = np.asarray(vert_variable2_n1)
                vert_variable3_n1 = np.asarray(vert_variable3_n1)
                vert_variable4_n1 = np.asarray(vert_variable4_n1)

                print("variable1_n1 = ", vert_variable1_n1)

                # if simulation == 'linear_simulation':
                # Well-posedness check
                i = 0
                for variable1_test, variable2_test, variable3_test, variable4_test in zip(vert_variable1_n1, vert_variable2_n1, vert_variable3_n1, vert_variable4_n1):
                    Aeval = np.asarray(
                        Amat(variable1_test, variable2_test, variable3_test, variable4_test))
                    Beval = np.asarray(
                        Bmat(variable1_test, variable2_test, variable3_test, variable4_test))

                    m_test, _ = eig(Beval, b=Aeval, overwrite_a=True,
                                    overwrite_b=True, check_finite=True)

                    listreal = []
                    listimag = []

                    for cond0 in range(len(m_test)):
                        realpart = m_test[cond0].real
                        imagpart = m_test[cond0].imag

                        listimag.append(imagpart)
                        listreal.append(realpart)

                    if any([cond1 != 0 for cond1 in listimag]):
                        print("INFO: ill-posed equations")
                        break
                    else:
                        i += 1
                    break

                # Condition for low liquid level of transition to intermittent flow
                if any([cond2 <= 0 for cond2 in vert_variable1_n1]):
                    print(
                        "INFO: gas phase dominant. Low liquid level. Liquid equations vanish")
                    break
                elif any([cond2 >= 1 for cond2 in vert_variable1_n1]):
                    print(
                        "INFO: liquid phase dominant. Intermittent flow. Gas equations vanish")
                    break

                # Condition for positive velocities and pressure
                if any([cond3 <= 0 for cond3 in vert_variable2_n1]):
                    print("INFO: negative liquid velocity")
                    break

                if any([cond4 <= 0 for cond4 in vert_variable3_n1]):
                    print("INFO: negative gas velocity")
                    break

                if any([cond5 <= 0 for cond5 in vert_variable4_n1]):
                    print("INFO: negative pressure")
                    break

            #     # Reynolds check
            #     Rel_n1 = Re_l (vert_variable1_n1, vert_variable2_n1)
            #     Reg_n1 = Re_g (vert_variable1_n1, vert_variable3_n1, vert_variable4_n1)

            #     if any ([ cond3 <= 1180 for cond3 in Rel_n1]):
            #     # Rel < 1180:
            #         print ("Liquid laminar")
            #     elif  any ([ cond3 > 1180 for cond3 in Rel_n1]):
            #         pass

            #     if any ([ cond4 <= 1180 for cond4 in Reg_n1]):
            #         print ("Gas laminar")
            #     elif  any ([ cond4 > 1180 for cond4 in Reg_n1]):
            #         pass

    # ===============================================================
    # END OF COMPUTATION FOR ALL TIME METHODS
    # ===============================================================
                # Plot solution var1
                plt.figure(5)
                plt.xlim(0, L)
                # plt.ylim (0, 1)
                matplotlib.rc('xtick', labelsize=label_size)
                matplotlib.rc('ytick', labelsize=label_size)
                plt.xlabel(r'L [m]', fontsize=label_size)
                plt.ylabel(r'$\alpha_l$ [-]', fontsize=label_size)
                plt.grid(True, which="both")
                # plt.legend(['Step2'])
                plot(variable1, label="step2", rescale=False, wireframe=False)

                # if T == t:
                #     # Plot figure
                #     fig.set_size_inches (mapsize)
                #     plt.savefig('results/figures/transient/fields/fig1.pdf',
                #                 optimize = True,
                #                 transparent = True,
                #                 dpi = dpi_elsevier)

                # Plot solution var2
                plt.figure(6)
                plt.xlim(0, L)
                # plt.ylim (min (vert_variable2_n1), max (vert_variable2_n1))
                matplotlib.rc('xtick', labelsize=label_size)
                matplotlib.rc('ytick', labelsize=label_size)
                plt.xlabel(r'L [m]', fontsize=label_size)
                plt.ylabel(r'$u_l$ [m/s]', fontsize=label_size)
                plt.grid(True, which="both")
                plot(variable2, rescale=False, wireframe=False)

                # if T == t:
                #     # Plot figure
                #     fig.set_size_inches (mapsize)
                #     plt.savefig('results/figures/transient/fields/fig2.pdf',
                #                 optimize = True,
                #                 transparent = True,
                #                 dpi = dpi_elsevier)

                # Plot solution var3
                plt.figure(7)
                plt.xlim(0, L)
                matplotlib.rc('xtick', labelsize=label_size)
                matplotlib.rc('ytick', labelsize=label_size)
                plt.xlabel(r'L [m]', fontsize=label_size)
                plt.ylabel(r'$u_g$ [m/s]', fontsize=label_size)
                plt.grid(True, which="both")
                plot(variable3, rescale=False, wireframe=False)

                # if T == t:
                #     # Plot figure
                #     fig.set_size_inches (mapsize)
                #     plt.savefig('results/figures/transient/fields/fig3.pdf',
                #                 optimize = True,
                #                 transparent = True,
                #                 dpi = dpi_elsevier)

                # Plot solution var4
                plt.figure(8)
                plt.xlim(0, L)
                matplotlib.rc('xtick', labelsize=label_size)
                matplotlib.rc('ytick', labelsize=label_size)
                plt.xlabel(r'L [m]', fontsize=label_size)
                plt.ylabel(r'$p_i$ [Pa]', fontsize=label_size)
                plt.grid(True, which="both")
                plot(variable4, rescale=False, wireframe=False)

                # if T == t:
                #     # Plot figure
                #     fig.set_size_inches (mapsize)
                #     plt.savefig('results/figures/transient/fields/fig4.pdf',
                #                 optimize = True,
                #                 transparent = True,
                #                 dpi = dpi_elsevier)

                # Save solution
                ff_variable1 << variable1
                ff_variable2 << variable2
                ff_variable3 << variable3
                ff_variable4 << variable4

                # Compute the amplification factor
                G1 = abs(np.divide(vert_variable1_n1, vert_variable1_n))
                G2 = abs(np.divide(vert_variable2_n1, vert_variable2_n))
                G3 = abs(np.divide(vert_variable3_n1, vert_variable3_n))
                G4 = abs(np.divide(vert_variable4_n1, vert_variable4_n))

                # print ("G1 = ", G1)
                # print ("G2 = ", G2)
                # print ("G3 = ", G3)
                # print ("G4 = ", G4)

                z1 = (a0*G1**2 + a1*G1 + a2)/(theta*G1**2 + (1 - theta)*G1)
                z2 = (a0*G2**2 + a1*G2 + a2)/(theta*G2**2 + (1 - theta)*G2)
                z3 = (a0*G3**2 + a1*G3 + a2)/(theta*G3**2 + (1 - theta)*G3)
                z4 = (a0*G4**2 + a1*G4 + a2)/(theta*G4**2 + (1 - theta)*G4)

                # print ("z1 = ", z1)
                # print ("z2 = ", z2)
                # print ("z3 = ", z3)
                # print ("z4 = ", z4)

                mu1 = z1/dt
                mu2 = z2/dt
                mu3 = z3/dt
                mu4 = z4/dt

                # print ("mu1 = ", mu1)
                # print ("mu2 = ", mu2)
                # print ("mu3 = ", mu3)
                # print ("mu4 = ", mu4)

                mu1_abs = abs(mu1)
                mu2_abs = abs(mu2)
                mu3_abs = abs(mu3)
                mu4_abs = abs(mu4)

                # print ("mu1_abs = ", mu1_abs)
                # print ("mu2_abs = ", mu2_abs)
                # print ("mu3_abs = ", mu3_abs)
                # print ("mu4_abs = ", mu4_abs)

                localmax_mu1 = np.where(mu1_abs == mu1_abs.max())
                localmax_mu2 = np.where(mu2_abs == mu2_abs.max())
                localmax_mu3 = np.where(mu3_abs == mu3_abs.max())
                localmax_mu4 = np.where(mu4_abs == mu4_abs.max())

                # print ("max mu1 position = ", localmax_mu1[0][0])
                # print ("max mu2 position = ", localmax_mu2[0][0])
                # print ("max mu3 position = ", localmax_mu3[0][0])
                # print ("max mu4 position = ", localmax_mu4[0][0])

                # print ("G1 for mumax = ", G1[localmax_mu1[0]])
                # print ("G2 for mumax = ", G2[localmax_mu2[0]])
                # print ("G3 for mumax = ", G3[localmax_mu3[0]])
                # print ("G4 for mumax = ", G4[localmax_mu4[0]])

                # numstab = z.real

                variable_n.vector()[:] = variable.vector()

        # Hold plot
        plt.show()

        if T == t:
            # L2 norm variation (Euclidean norm)
            if num_steps > 1:

                # Plot L2 norm for var1
                fig, ax = plt.subplots()
                # plt.rcParams ['figure.figsize'] = mapsize
                ax.plot(timevector,
                        L2norm_variable1,
                        '-k')
                ax.set_xlabel('Time [s]', fontsize=label_size)
                ax.set_ylabel(r"$L^2$ norm of $\alpha_l$", fontsize=label_size)
                plt.xlim(0, T)
                plt.grid(True,
                         which="both")

                matplotlib.rc('xtick', labelsize=label_size)
                matplotlib.rc('ytick', labelsize=label_size)

                # leg = ax.legend (loc = 'best',
                #                 shadow = True,
                #                 frameon = True)

                # Plot figure
                fig.set_size_inches(mapsize)
                plt.savefig('results/figures/transient/l2norm/fig1.pdf',
                            optimize=True,
                            transparent=True,
                            dpi=dpi_elsevier)

                # Show plot
                plt.show()

            # Plot L2 norm for var2
                fig, ax = plt.subplots()
                # plt.rcParams ['figure.figsize'] = mapsize
                ax.plot(timevector,
                        L2norm_variable2,
                        '-k')
                ax.set_xlabel('Time [s]', fontsize=label_size)
                ax.set_ylabel(r"$L^2$ norm of $u_l$", fontsize=label_size)
                plt.xlim(0, T)
                plt.grid(True,
                         which="both")

                matplotlib.rc('xtick', labelsize=label_size)
                matplotlib.rc('ytick', labelsize=label_size)

                # leg = ax.legend (loc = 'best',
                #                 shadow = True,
                #                 frameon = True)

                # Plot figure
                fig.set_size_inches(mapsize)
                plt.savefig('results/figures/transient/l2norm/fig2.pdf',
                            optimize=True,
                            transparent=True,
                            dpi=dpi_elsevier)

                # Show plot
                plt.show()

            # Plot L2 norm for var3
                fig, ax = plt.subplots()
                # plt.rcParams ['figure.figsize'] = mapsize
                ax.plot(timevector,
                        L2norm_variable3,
                        '-k')
                ax.set_xlabel('Time [s]', fontsize=label_size)
                ax.set_ylabel(r"$L^2$ norm of $u_g$", fontsize=label_size)
                plt.xlim(0, T)
                plt.grid(True,
                         which="both")

                matplotlib.rc('xtick', labelsize=label_size)
                matplotlib.rc('ytick', labelsize=label_size)

                # leg = ax.legend (loc = 'best',
                # shadow = True,
                # frameon = True)

                # Plot figure
                fig.set_size_inches(mapsize)
                plt.savefig('results/figures/transient/l2norm/fig3.pdf',
                            optimize=True,
                            transparent=True,
                            dpi=dpi_elsevier)

                # Show plot
                plt.show()

            # Plot L2 norm for var4
                fig, ax = plt.subplots()
                # plt.rcParams ['figure.figsize'] = mapsize
                ax.plot(timevector,
                        L2norm_variable4,
                        '-k')
                ax.set_xlabel('Time [s]', fontsize=label_size)
                ax.set_ylabel(r"$L^2$ norm of $p_i$", fontsize=label_size)
                plt.xlim(0, T)
                plt.grid(True,
                         which="both")

                matplotlib.rc('xtick', labelsize=label_size)
                matplotlib.rc('ytick', labelsize=label_size)

                # leg = ax.legend (loc = 'best',
                #                 shadow = True,
                #                 frameon = True)

                # Plot figure
                fig.set_size_inches(mapsize)
                plt.savefig('results/figures/transient/l2norm/fig4.pdf',
                            optimize=True,
                            transparent=True,
                            dpi=dpi_elsevier)
                # Show plot
                plt.show()

    # ===============================================================
    # EIGENSPECTRUM
    # ===============================================================
            if transient_eigenspectrum == 1:
                # create vectors
                maprealeig = [[] for i in range(len(vert_variable1_n1))]
                mapimageig = [[] for i in range(len(vert_variable1_n1))]

                list_unstable = []

                i = 0
                for variable1_test, variable2_test, variable3_test, variable4_test in zip(vert_variable1_n1, vert_variable2_n1, vert_variable3_n1, vert_variable4_n1):
                    Aeval_sp, Beval_sp, Ceval_sp = linear_matrices_function(
                        variable1_test, variable2_test, variable3_test, variable4_test, beta, D, system, rho_l, p_factor, mu_l, mu_g, dirichlet_type)

                    listreal2_sp, listimag2_sp, m2_sp, _ = stability_function(
                        Aeval_sp, Beval_sp, Ceval_sp)

                    listreal_eig = []
                    listimag_eig = []

                    for cond8 in range(len(m2_sp)):
                        # realpart_eig = m2_sp [cond8].real
                        # imagpart_eig = m2_sp [cond8].imag

                        listimag_eig.append(listimag2_sp)
                        listreal_eig.append(listreal2_sp)

                        mapimageig[i] = np.array(listimag_eig)
                        maprealeig[i] = np.array(listreal_eig)
                    i += 1

                fig, ax = plt.subplots()

                for ii in range(len(vert_variable1_n1)):
                    ax.scatter(maprealeig[ii],
                               -mapimageig[ii],
                               s=area_scatter,
                               marker=listmarkers[0],
                               color=listcolor[4],
                               edgecolors=listcolor[0],
                               linewidths=line_width,
                               alpha=alphascatter)
                    plt.grid(True, which="both")
                    # ax.set_xscale ('symlog')
                    # plt.rcParams ['figure.figsize'] = mapsize
                    matplotlib.rc('xtick', labelsize=label_size)
                    matplotlib.rc('ytick', labelsize=label_size)
                    # plt.ylim (-10, 10)
                    # # plt.xlim (-1e3, 1e3)
                    ax.xaxis.set_tick_params(
                        which='major', direction='in', top='on')
                    ax.xaxis.set_tick_params(
                        which='minor', direction='in', top='on')
                    ax.yaxis.set_tick_params(
                        which='major', direction='in', right='on')
                    ax.yaxis.set_tick_params(
                        which='minor', direction='in', right='on')

                    plt.xlabel(
                        r'Re ($\mu$) $[\it{s^{-1}}]$', fontsize=label_size)
                    plt.ylabel(
                        r'Im ($\mu$) $[\it{s^{-1}}]$', fontsize=label_size)

                    fig.set_size_inches(mapsize)
                    plt.savefig('results/figures/transient/eigenspectrum/fig1.pdf',
                                optimize=True,
                                transparent=True,
                                dpi=dpi_elsevier)

                    # if any ([ cond1 < 0 for cond1 in mapimageig[i]]) :
                    #     # print ("INFO: unstable equation system")
                    #     # print (min (mapimageig[i]))
                    #     list_unstable.append (min (mapimageig[i]))
                    # else:
                    #     pass
                    # print ("INFO: stable equation system")

                # list_unstable = np.array (list_unstable)
                # if all ([ cond1 < 0 for cond1 in list_unstable]) :
                #     print (np.array(list_unstable))

                    # fig, ax = plt.subplots ()
                    # for ii in range (len (vert_variable1_n1)):
                    #     ax.scatter (maprealeig[ii],
                    #         -mapimageig[ii],
                    #         s = area,
                    #         marker = listmarkers [0],
                    #         color = listcolor [4],
                    #         edgecolors = listcolor [0],
                    #         linewidths = 1.5,
                    #         alpha = 0.5)
                    #     plt.grid (True, which = "both")
                    #     # ax.set_xscale ('symlog')
                    #     plt.rcParams ['figure.figsize'] = [12, 8]
                    #     matplotlib.rc ('xtick', labelsize = 14)
                    #     matplotlib.rc ('ytick', labelsize = 14)
                    #     # plt.ylim (-10, 10)
                    #     # # plt.xlim (-1e3, 1e3)
                    #     ax.xaxis.set_tick_params(which='major', size=10, direction='in', top='on')
                    #     ax.xaxis.set_tick_params(which='minor', size=7, direction='in', top='on')
                    #     ax.yaxis.set_tick_params(which='major', size=10, direction='in', right='on')
                    #     ax.yaxis.set_tick_params(which='minor', size=7, direction='in', right='on')
                    #     ax.set_xlabel(r'Re [$\lambda$] [1/s]', fontsize = 18)
                    #     ax.set_ylabel(r'-Im [$\lambda$] [1/s]', fontsize = 18)

# (base) root@MacBook twofluidmodel # conda activate fenicsproject
# (fenicsproject) root@MacBook twofluidmodel # ./modules/fully_discretized.py
