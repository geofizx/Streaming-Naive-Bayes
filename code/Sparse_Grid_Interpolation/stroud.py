#!/usr/bin/env python2.7
# encoding: utf-8

"""
Stroud/Xiu Sparse Samplers

Compute stochastic collocation node points based on Stroud-2, Stroud-3, 
Xiu-2, or Xiu-3 cubature formulas of corresponding degrees:

ndim : integer - number of random vectors to compute grids
soln : integer - Which formula to use:
  	(2) : Stroud deg-2 - uniform priors
  	(3) : Stroud deg-3 - uniform priors
 	(4) : Xiu deg-2 - Gaussian priors
 	(5) : Xiu deg-3 - Gaussian priors

Stroud Formulas are exact for symmetric uniform distribution pdfs
For Stroud Formulas -See Stroud, 1957, "Remarks on the Disposition of
Points in Numerical Integration Formula", MTAC, v.11, p. 257.

Xiu Formulas are exact for symmetric Gaussian pdfs
For Xiu Formulas - See Xiu, 2007, "Numerical Integration Formulas of
Degree 2"

@author Michael Tompkins
@copyright 2016
"""

#Externals
import numpy as npy
import math

def stroud(N,soln): 
	
	if soln == 2 : #Use Stroud degree-2 cubature nodes for Uniform Priors

		Y = npy.ndarray(shape=(N+1,N),dtype=float)

		if math.floor(N/2) == N/2.0:
			for k in range(0,N+1):
				for r in range(1,(N/2)+1):
					i = r
					Y[k,2*i-2] = npy.sqrt(2./3.)*(math.cos((2*r*k*math.pi)/(N+1)))
					Y[k,2*i-1] =   npy.sqrt(2./3.)*(math.sin((2*r*k*math.pi)/(N+1)))

		else:
				for k in range(0,N+1):
					for r in range(1,((N-1)/2)+1):
							i = r
							Y[k,2*i-2] = npy.sqrt(2./3.)*(math.cos((2*r*k*math.pi)/(N+1)))
							Y[k,2*i-1] =   npy.sqrt(2./3.)*(math.sin((2*r*k*math.pi)/(N+1)))
					#Employ odd formula for last node point
					Y[k,N-1] = ((-1)**k)/npy.sqrt(3.)

	elif soln == 3 : #Use Stroud degree-3 cubature nodes for Uniform Priors

		Y = npy.ndarray(shape=(N*2,N),dtype=float)

		if math.floor(N/2) == N/2.0:
				for k in range(0,2*N+1):
					for r in range(1,(N/2)+1):
							i = r
							Y[k-1,2*i-2] = npy.sqrt(2./3.)*(math.cos(((2*r-1)*k*math.pi)/(N)))
							Y[k-1,2*i-1] =   npy.sqrt(2./3.)*(math.sin(((2*r-1)*k*math.pi)/(N)))

		else:
				for k in range(0,2*N+1):
					for r in range(1,((N-1)/2)+1):
							i = r
							Y[k-1,2*i-2] = npy.sqrt(2./3.)*(math.cos(((2*r-1)*k*math.pi)/(N)))
							Y[k-1,2*i-1] =   npy.sqrt(2./3.)*(math.sin(((2*r-1)*k*math.pi)/(N)))
					#Employ odd formula for last node point
					Y[k-1,N-1] = ((-1)**k)/npy.sqrt(3.)

	elif soln == 4: #Use Xiu-2 cubature nodes for Gaussian Priors

		Y = npy.ndarray(shape=(N+1,N),dtype=float)

		if math.floor(N/2) == N/2.0:
				for k in range(0,N+1):
					for r in range(1,(N/2)+1):
							i = r
							Y[k,2*i-2] = npy.sqrt(2.)*(math.cos((2*r*k*math.pi)/(N+1)))
							Y[k,2*i-1] =   npy.sqrt(2.)*(math.sin((2*r*k*math.pi)/(N+1)))

		else:
				for k in range(0,N+1):
					for r in range(1,((N-1)/2)+1):
							i = r
							Y[k,2*i-2] = npy.sqrt(2.)*(math.cos((2*r*k*math.pi)/(N+1)))
							Y[k,2*i-1] =   npy.sqrt(2.)*(math.sin((2*r*k*math.pi)/(N+1)))
					#Employ odd formula for last node point
					Y[k,N-1] = ((-1)**k)

	elif soln == 5: #Use Xiu degree-3 cubature nodes for Gaussian Priors

		Y = npy.ndarray(shape=(N*2,N),dtype=float)

		if math.floor(N/2) == N/2.0:
				for k in range(0,2*N+1):
					for r in range(1,(N/2)+1):
							i = r
							Y[k-1,2*i-2] = npy.sqrt(2.)*(math.cos(((2*r-1)*k*math.pi)/(N)))
							Y[k-1,2*i-1] =   npy.sqrt(2.)*(math.sin(((2*r-1)*k*math.pi)/(N)))

		else:
				for k in range(0,2*N+1):
					for r in range(1,((N-1)/2)+1):
							i = r
							Y[k-1,2*i-2] = npy.sqrt(2.)*(math.cos(((2*r-1)*k*math.pi)/(N)))
							Y[k-1,2*i-1] =   npy.sqrt(2.)*(math.sin(((2*r-1)*k*math.pi)/(N)))
					#Employ odd formula for last node point
					Y[k-1,N-1] = ((-1)**k)

	else:
		raise Exception("Solution type must be 2-5")

	return Y


