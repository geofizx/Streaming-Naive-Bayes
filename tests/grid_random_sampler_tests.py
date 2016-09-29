#!/usr/bin/env python2.7
# encoding: utf-8

"""
Some unit tests and usage examples for random and sparse grid sampler class

@Usage Examples and Tests

2D Chebyshev sparse grid nodes
2D Clenshaw-Curtis sparse grid nodes
2D Poisson Disk samples
2D Uniform Random samples

Generate plots for some outputs

@author Michael Tompkins
@copyright 2016
"""

import matplotlib.pyplot as pl
from Sparse_Grid_Interpolation import ndSampler

# Determine which tests will be run with bools
Poisson = True
Chebyshev = True
Clenshaw = True
Uniform = True

if Poisson is True:
	num = 400		# Number of samples to draw
	dim1 = 2		# Dimensionality of space
	sample = ndSampler(num/2,dim1)
	candidates = 20	# Number of candidate samples for each numsim iteration of sampler
	points1 = sample.poissondisk(candidates)
	sample = ndSampler(num,dim1)
	points2 = sample.poissondisk(candidates)
	label1 = str(num/2)+" Samples"
	label2 = str(num)+" Samples"
	pl.plot(points1[:,0],points1[:,1],'ro',label=label1)
	pl.hold(True)
	pl.plot(points2[:,0],points2[:,1],'bo',label=label2)
	pl.title("Poisson Disk Random Samples")
	pl.legend()
	pl.show()

if Chebyshev is True:
	num = 4			# Degree of polynomial to compute
	dim1 = 2		# Dimensionality of space
	sample = ndSampler(num,dim1)
	points1,mi0,indx0 = sample.sparse_sample("CH")
	sample = ndSampler(num/2,dim1)
	points2,mi0,indx0 = sample.sparse_sample("CH")
	label1 = "Degree:"+str(num)+" Nodes"
	label2 = "Degree:"+str(num/2)+" Nodes"
	pl.plot(points1[:,0],points1[:,1],'ro',label=label1)
	pl.hold(True)
	pl.plot(points2[:,0],points2[:,1],'bo',label=label2)
	pl.title("Chebyshev Sparse-Grid Samples")
	pl.legend()
	pl.show()

if Clenshaw is True:
	num = 4			# Degree of polynomial to compute
	dim1 = 2		# Dimensionality of space
	sample = ndSampler(num,dim1)
	points1,mi0,indx0 = sample.sparse_sample("CC")
	sample = ndSampler(num/2,dim1)
	points2,mi0,indx0 = sample.sparse_sample("CC")
	label1 = "Degree:"+str(num)+" Nodes"
	label2 = "Degree:"+str(num/2)+" Nodes"
	pl.plot(points1[:,0],points1[:,1],'ro',label=label1)
	pl.hold(True)
	pl.plot(points2[:,0],points2[:,1],'bo',label=label2)
	pl.title("Clenshaw-Curtis Sparse-Grid Samples")
	pl.legend()
	pl.show()

if Uniform is True:
	num = 400		# Number of samples to draw
	dim1 = 2		# Dimensionality of space
	sample = ndSampler(num/2,dim1)
	points1 = sample.unfrm()
	sample = ndSampler(num,dim1)
	points2 = sample.unfrm()
	label1 = str(num/2)+" Samples"
	label2 = str(num)+" Samples"

	pl.plot(points1[:,0],points1[:,1],'ro',label=label1)
	pl.hold(True)
	pl.plot(points2[:,0],points2[:,1],'bo',label=label2)
	pl.title("Uniform Random Samples")
	pl.legend()
	pl.show()