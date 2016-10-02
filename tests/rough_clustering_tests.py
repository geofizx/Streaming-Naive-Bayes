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
import matplotlib.pyplot as plt
import numpy as npy
from scipy.cluster.vq import kmeans2

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

# Compare results with known centroid mean and std deviations as well as those from k-means

# Print stats for members of clusters
# Determine labels
list1 = [i for i in range(len(data["response"])) if data["response"][i] == '1']
list2 = [i for i in range(len(data["response"])) if data["response"][i] == '2']

tableau_lists = []
tableau_1 = []
tableau_2 = []
for key in header:
	tableau_lists.append(data2[key][:])
	tableau_1.append(npy.asarray(data2[key])[list1].tolist())
	tableau_2.append(npy.asarray(data2[key])[list2].tolist())
datav = npy.asfarray(tableau_lists).T
data1 = npy.asarray(tableau_1).T
data3 = npy.asarray(tableau_2).T
plt.scatter(npy.squeeze(data1[:,2]),npy.squeeze(data1[:,13]),c="b",label="Good")
plt.hold(True)
plt.scatter(npy.squeeze(data3[:,2]),npy.squeeze(data3[:,13]),c="r",label="Bad")
plt.title("Checking Account versus Account Duration for Credit Risk")
plt.show()

mean1 = npy.mean(data1,axis=0)
mean2 = npy.mean(data3,axis=0)
std1 = npy.std(data1,axis=0)
std2 = npy.std(data3,axis=0)

# Just run k-means to compare
[centroids,groups] = kmeans2(datav,2,iter=20)
meank = [[] for g in range(2)]
val = [[] for n in range(len(groups))]
for m in range(len(groups)):
	for n in range(2):
		if groups[m] == n:
			val[n].append(data["response"][m])
			meank[n].append(datav[int(val[n][-1]),:])
meankp = []
stddevk = []
for n in range(2):
	print Counter(val[n])
	meankp.append(npy.mean(meank[n],axis=0))
	stddevk.append(npy.std(meank[n],axis=0))

# Compile stats for rough clusters
resultsm = []
resultss = []
rangek = [l+.2 for l in range(20)]
ranger = [l+.1 for l in range(20)]
print "total instances",Counter(data["response"])

key1 = 17 # Which distance D to plot
fig, axs = plt.subplots(nrows=1,ncols=1)
axs.errorbar(range(20),mean2,fmt='ro',yerr=std2,label="True Good Centroid")
plt.hold(True)
axs.errorbar(range(20),mean1,fmt='bo',yerr=std1,label="True Bad Centroid")
axs.errorbar(rangek,meankp[1],fmt='r+',yerr=stddevk[0],label="Kmeans 0")
axs.errorbar(rangek,meankp[0],fmt='b+',yerr=stddevk[1],label="Kmeans 1")

for key in clust.pruned[key1]["cluster_list"][0]:
	print key,clust.pruned[key1]["cluster_list"][0].keys()
	print clust.pruned[key1]["cluster_list"][0][key]
	datav2 = []
	meant = []
	stdt = []
	for val in clust.pruned[key1]["cluster_list"][0][key]:
		meant.append(data["response"][int(val)])
		datav2.append(datav[int(val),:])
	tmp = npy.mean(npy.asarray(datav2),axis=0)
	tmp2 = npy.std(npy.asarray(datav2),axis=0)
	if key1 == 17:
		if key == 13:
			axs.errorbar(ranger,tmp,fmt='rv',yerr=tmp2,label=str(key1)+" "+str(key))
		if key == 14:
			axs.errorbar(ranger,tmp,fmt='bv',yerr=tmp2,label=str(key1)+" "+str(key))
	else:
		axs.errorbar(ranger,tmp,yerr=tmp2,label=str(key1)+" "+str(key))
		resultsm.append(npy.mean(npy.asarray(datav2),axis=0))
		resultss.append(npy.std((npy.asarray(datav2)),axis=0))
		print key1,key,len(meant),Counter(meant)
plt.legend()
plt.show()
