#!/usr/bin/env python2.7
# encoding: utf-8

"""
Description Here

Created by Michael Tompkins on 201x-xx-xx.
Copyright (c) 201x__PolytopX__. All rights reserved.
"""

#Externals
import json
from collections import Counter
from Machine_Learning import roughCluster

# Set some rough clustering parameters
maxD = 18
max_clusters = [2]

# Load some data
file2 = open("german_all.json","r")
data = json.load(file2)
print data.keys()

# Do some numerical encoding for input payload
header = []
#header = ['checking', 'duration', 'history', 'purpose', 'amount', 'savings', 'employment', 'rate',
#		  'status_sex', 'guarantors', 'residence', 'property', 'age', 'other_plans', 'housing',
#		  'existing_credit', 'job_status', 'liable', 'telephone', 'foreign_worker']
data2 = {}
for key in data["payload"].keys():
	header.append(key)
	#print key,len(Counter(data["payload"][key]).keys()),Counter(data["payload"][key]).keys()
	try:
		data2[key] = [int(data["payload"][key][m]) for m in range(0,len(data["payload"][key]))]
		if key == "amount":
			data2[key] = []
			for n in range(len(data["payload"][key])):
			#[values,binsf] = npy.histogram(data2[key],[0,1500,3000,8000,20000])
				bins = [0,1500,3000,8000,20000]
				for i,val in enumerate(bins[0:-1]):
					if (int(data["payload"][key][n])) >= val and (int(data["payload"][key][n]) < bins[i+1]):
						data2[key].append(i+1)
	except:
		data2[key] = []
		encoding = {key : m for m,key in enumerate(Counter(data["payload"][key]).keys())}
		#print key,encoding
		for n in range(len(data["payload"][key])):
			data2[key].append(encoding[data["payload"][key][n]])

# Instantiate and run rough clustering
clust = roughCluster(data2,maxD,max_clusters)
clust.enumerateClusters()
clust.optimizeClusters()
print clust.clusters
print clust.sum_upper
print clust.sum_lower
