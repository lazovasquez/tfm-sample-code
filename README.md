Two-fluid model sample code for JBSMSE
============================================

Description 
-----------

This repository contains a unified framework for performing transient simulations and stability analysis of two-phase flows using the linearized one-dimensional averaged Navier-Stokes equations. The code uses the finite element method for constructing fully-discrete flow pattern maps by resolving the linearized equations and performing linear stability analysis. Also, the capability of the numerical formulation for describing wave growth is accomplished through stiffness analysis of the semi-discrete equations.

After formulating the variational problem, the [FEniCS Project](https://fenicsproject.org/) library is employed to construct the basis functions, solve the integral functions on the element level, and assemble the system matrix. The system matrix is solved through the Newton method and the eigenvalue problem through the ARPACK library. The code is written in the Python 3 language.

Requirements
------------

The code requires Python 3.6 or higher, NumPy, SciPy, Matplotlib, math, and FEniCS.
