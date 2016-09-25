#!/usr/bin/env python2.7
# encoding: utf-8

"""
@desciption

Class to perform hierarchical sparse-grid polynomial interpolation at multiple grid levels using
either piece-wise linear (type = 'CC') or Chebyshev polynomial (type = 'CH') basis functions at
sparse grid nodes specified by max degree (i.e., level) of interpolation and dimensionality of space.

Early stopping is implemented when absolute error at any level is less than tol

@usage
[ip2,wk,werr,meanerr] = mjvals(maxn,d,type,intvl,fun_nd,grdout)

	:arg

	maxn : integer : maxlevel of interpolation depth
	d : integer : dimension of interpolation
	type : string : polynomial : piece-wise linear (type = 'CC') or Chebyshev polynomial (type = 'CH') to use for interpolation
	intvl : 2 x d array : interval other than [0 1] for each dimension over which to compute sparse grids
	fun_nd : function : user-defined function used to evaluate the target function at sparse grid nodes
	grdout : N x d array : desired N points in d-dimensions to interpolate to

	:return
	ip2 : N-length array of interpolated values on the user specified grid grdout
	werr : absolute error of interpolation for each level [0-maxn]
	meanerr : mean error of interpolation for each level [0-maxn]
	wk : hiearchical surpluses for each interpolation level [0-maxn]

@dependencies
mjgrid.m - Companion script for computing sparse grid node points
mjnterp.m - Companion script for computing hierarchical surpluses

This script also calls function rmsint.m which queries "fun_nd" for the
RMS misfit values for models at sparse grid points [grdin]

@references
See Klemke, A. and B. Wohlmuth, 2005, Algorithm 847: spinterp: Piecewise
Multilinear Hierarchical Sparse Grid Interpolation in MATLAB,
ACM Trans. Math Soft., 561-579.

@author Michael Tompkins in 2012.
@copywrite (c) 2014 All rights reserved.
"""

import numpy as npy
import samplers as samplers
import spinterp
from copy import deepcopy
import fun_nd

class sparseInterp():

	def __init__(self, maxn, dimensions, grdout, type, intvl=None):

		"""
		:arg numsim : integer number of points to draw
		:arg d : integer dimensionality of sampling space
		:return : Y : array-type (numsim,d) of numsim sampled points in d-dimensions

		options - verbose : turn on (True) or off (False) print statements
		"""

		self.grdout = grdout
		self.maxn = maxn				# Maximum degree of interpolation to perform -- see self.tol for early stopping
		self.d = dimensions				# Number of dimensions of interpolation
		self.dim1 = grdout.shape[0]		# Number of samples for output
		self.dim2 = grdout.shape[1]		# Dimensionality of interpolation
		self.verbose = False			# Include print statements to stdout
		self.debug = 0					# 0 = user defined function used, 1 = unit test fun2d used
		self.tol = 0.001				# Early stopping criteria
		self.type = type				# Type of polynomial to perform interpolation
		self.intvl = intvl				# Interval over which to perform interpolation

	def runInterp(self):

		"""
		Perform n-d sparse grid interpolation
		"""

		grdout = self.grdout
		type = self.type
		intvl = self.intvl
		maxn = self.maxn
		d = self.d
		debug = self.debug
		num2 = self.dim1								# Number of points to interpolate on user input grid
		ip2 = npy.zeros(shape=(num2),dtype=float)		# Initialize final interpolated array
		ipmj = npy.zeros(shape=(num2),dtype=float)		# Initialize d-variate interpolant array
		tol = self.tol									# Early stopping criteria for interpolation

		grdbck = {}		# Dictionary for back storage of grid arrays at each level k, for hierarchical error checking
		indxbck = {}	# Dictionary for back storage of grid index arrays at each level k, for hierarchical error checking
		mibck = {}		# Dictionary for back storage of grid sample # at each level k, for hierarchical error checking
		yk = {}			# Dictionary of function evaluations for each level k
		wk = {}			# Dictionary of hierarchical surpluses for each level k
		meanerr = {}	# Dictionary of hierarchical surpluses mean errors for each level k
		werr = {}		# Dictionary of hierarchical surpluses absolute errors for each level k

		# TODO need to check that fun_nd is valid function evalution and exists
		# Loop over all grid levels (i.e., polynomial degree) from k=0:maxn to determine optimal level for interpolation
		# Break criteria is when mx{zk} <= toler
		for k in xrange(0,maxn+1):

			"""
			Determine index sets and sparse grid nodes for each grid level k and dimension d, type of interpolation,
			and interval [0 1]
			See <help mjgrid> for description of these parameters
			#[grdin,indx,mi] = mjgrid(k,d,type,[0,1])
			"""

			samp = samplers.ndSampler(k,d)		# Instantitate sampler class for current level k (degree) of interpolation
			grdin,mi,indx = samp.cheby()		# Compute polynomial nodes for each level k of interpolation

			if k == 0:
				indx = indx[npy.newaxis,:]		# Add dimension when 1-D array returned for level 0 grid

			# Stretch/squeeze grid to interval [intvl] in each dimension
			for i in xrange(0,d):
				range1 = abs( min(intvl[:,i]) - max(intvl[:,i]) )
				grdin[:,i] = grdin[:,i]*range1 + min(intvl[:,i])

			grdbck[k] = grdin          # Back storage of grid k, for access later
			indxbck[k] = indx          # Back storage of index array at level k
			mibck[k] = mi              # Back storage of node number array
			num4 = indx.shape[0]		# # of multi-indices to compute cartesian products
			wght = npy.zeros(shape=(num4,d),dtype=float) # Initialize weights for current level k
			wght2 = npy.ones(shape=(num4),dtype=float)
			polyw = npy.zeros(shape=(num4,d),dtype=float)

			# Determine function values at current kth sparse grid nodes using user-defined function fun_nd
			yk[k] = fun_nd.fun_nd(grdin)

			# Initialize surpluses to current grid node values
			zk = yk[k]

			# Now compute hierarchical surpluses by subtracting interpolated values
			# of current grid nodes (interp(grdin)) computed at grid level k-1 interpolant
			# from current function values (fun(x)) computed at current grid level, k.
			# e.g., zk(@ k=2) = fun(grdin, @k=2) - interp(grdin, @k=1)
			#
			# This allows for the determination of error at each grid level and a simpler implementation
			# of the muti-variate interpolation at various Smoyak grid levels.
			if k > 0:
				if werr[k-1] < tol:       # Stop criteria based on average surplus error
					print 'Mean error tolerance met at Grid Level...',str(k-1)
					return ip2,meanerr,werr
				else:
					for m in range(0,k): #i=0:k-1          # Must loop over all levels to get complete interpolation (@ k-1)
						runterp = spinterp.runInterp(d,wk[m],grdbck[m],grdin,indxbck[m],mibck[m],type,intvl)
						zk -= runterp

			# Loop over indices and grid points to perform interpolate at current grid level, k, using surpluses, zk.

			# Formulas based on Clenshaw-Curtis piecewise multi-linear basis functions of the kind:
			# wght2_j = 1-(mi-1)*norm(x-x_j), if norm(x-x_j)> 1/(mi-1), else wght2_j = 0.0
			if type == 'cc' or type == 'CC':
				for i in range(0,num2):			# Number of points to interpolate
					for j in range(0,num4):		# Number of grid nodes at current level
						wght2[j] = 1.0			# Iinitialize total linear basis integration weights
						for l in range(0,d): 	# Number of dimensions for the interpolation
							# Determine 1D linear basis functions for each index i
							if mi[indx[j,l]] == 1:
								wght[j,l] = 1.0	# Leave weight == 1.0 if mi = 1
							elif npy.linalg.norm(grdout[i,l]-grdin[j,l]) < (1./(mi[indx[j,l]]-1)):
								wght[j,l] = 1-(mi[indx[j,l]]-1)*npy.linalg.norm(grdout[i,l]-grdin[j,l])   # Compute 1D linear basis functions
							else:
								wght[j,l] = 0.0			# Compute 1D linear basis functions
							wght2[j] *= wght[j,l]		# Perform the dimensional products for the basis functions
						ipmj[i] += wght2[j]*zk[j]		# Sum over the number of total node points (j=num4) for all dimensions
					ip2[i] = ipmj[i] 					# Now re-assign the interpolated value to new variable (redundant)

			# Formulas based on Barycentric Chebyshev polynomial basis functions of the kind:
			# wght2_j = SUM_x_m[(x - x_m)/(x_j - x_m)], for all x_m != x_j
			elif type == 'ch' or type == 'CH':
				for i in range(0,num2):				# Number of points to interpolate
					for j in range(0,num4):			# Number of grid nodes at current level
						wght2[j] = 1.0				# Initialize total Chebyshev integration weights (i.e., w(x))
						for l in range(0,d):		# Number of dimensions for the interpolation
							polyw[j,l] = 1.0		# Iinitialize d-dim polynomial (i.e., (x - x_m)/(x_m - x_j))
							if mi[indx[j,l]] != 1:	# Leave weight == 1.0 if mi = 1
								for m in range(0,mi[indx[j,l]]):		#m=1:mi(indx(j,l)):   # Else compute weight products over number of nodes for a given mi
									xtmp = (1.+(-npy.cos((npy.pi*(m))/(mi[indx[j,l]]-1))))/2.    # Compute 1D node position on-the-fly
									# Transform xtmp based on interval
									range1 = npy.abs(npy.min(intvl[:,l])-npy.max(intvl[:,l]) )
									xtmp = xtmp*range1 + npy.min(intvl[:,l])
									if npy.abs(grdin[j,l] - xtmp) > 1.0e-03:	# Polynomial not defined if xtmp==grdin(j,l)
										polyw[j,l] = polyw[j,l]*( (grdout[i,l]-xtmp)/(grdin[j,l]-xtmp) )   # Perform 1D polynomial products
							wght2[j] *= polyw[j,l]         # Perform the dimensional products for the polynomials
						ipmj[i] += wght2[j]*zk[j]			# Sum over the number of total node points (j=num4) for all dimensions
					ip2[i] = ipmj[i]                               # Now re-assign the interpolated value to new variable (redundant)
				print k,i,j

			else:
				print 'error: type must be "cc" or "ch"'
				exit(1)

			wk[k] = zk                       	   # Assign current surpluses to wk for back storage and output
			werr[k] = npy.max(npy.abs(wk[k]))      # Compute absolute error of current grid level
			meanerr[k] = npy.mean(npy.abs(wk[k]))  # Compute mean error of current grid level

		ip2 = ip2.T									# Take the transpose of the interpolated vector output to conform

		return ip2,meanerr,werr

if __name__ == "__main__":

	"""
	Run unit tests
	"""

	import matplotlib.pyplot as pl
	import fun_nd
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm

	n = 4
	dim1 = 2
	gridout = npy.asarray([[0.0,0.25,0.5,0.75,1.0],[0.0,0.25,0.5,0.75,1.0]]).T
	[xx,yy] = npy.meshgrid([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	gridout = npy.asarray([xx.reshape(121),yy.reshape(121)]).T
	intval = npy.asarray([[0.0,1.0],[0.0,1.0]]).T
	type1 = "CC"

	# Instantiate and run interpolation for Chebyshev Polynomials
	interp = sparseInterp(n, dim1, gridout, type1, intval)
	output,meanerr1,werr1 = interp.runInterp()

	# Compare results with true function
	tmpvals = npy.asarray(fun_nd.fun_nd(gridout))
	tmpval2 = tmpvals.reshape(11,11)

	fig = pl.figure()
	ax = fig.add_subplot(131, projection='3d',title="True Function")
	ax.plot_surface(xx, yy, tmpval2,  rstride=1, cstride=1, cmap=cm.jet)
	ax = fig.add_subplot(132, projection='3d',title="Interpolation")
	tmpval3 = output.reshape(11,11)
	ax.plot_surface(xx, yy, tmpval3,  rstride=1, cstride=1, cmap=cm.jet)
	ax = fig.add_subplot(133, projection='3d', title="Interpolation Error")
	tmpval4 = npy.abs(tmpval3 - tmpval2)
	ax.plot_surface(xx, yy, tmpval4,  rstride=1, cstride=1, cmap=cm.jet)
	ax.set_zlim(0.0,npy.max(tmpval4)*2)
	pl.show()

	print "Mean Error for Each Degree of Total Degree:",n,": ",meanerr1

