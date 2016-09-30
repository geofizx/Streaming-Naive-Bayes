#!/usr/bin/env python2.7
# encoding: utf-8

"""
An implementation of rough clustering using rough set theory and the algorithm outlined by Voges, Pope & Brown, 2002,
"Cluster Analysis of Marketing Data Examining On-line Shopping Orientation: A Comparison of k-means and
Rough Clustering Approaches"

@author Michael Tompkins
@copyright 2016
"""

#Externals
import json
import itertools
import operator
from collections import Counter
import numpy as npy
import clustering
import matplotlib.pyplot as plt
import featureSelection

#Internals

class roughCluster():

	def __init__(self,input_data,max_d=None,max_clusters=None):

		self.data = input_data
		self.debug = True
		self.small = 1.0e-10

		if max_d is None:
			self.maxD = 18	# Maximum entity distance for stopping further clustering, if maxD == number of instances, all identities are in all clusters
		else:
			self.maxD = max_d
		if max_clusters is None:
			self.max_clusters = [2,3,5,10]	# Used for rejection of clusters when maximizing the sum of lower approximations for all clusters
		else:
			self.max_clusters = max_clusters

	def enumerateClusters(self):

		# Compute distance matrix for all pairs
		data_length = len(self.data[header[0]])
		print "Data Length",data_length
		data["distance"] = {}
		for k in range(0,data_length):
			data["distance"][str(k)] = {str(j) : sum([abs(self.data[val][k]-self.data[val][j]) for val in header])
										for j in range(0,data_length)}

		# Form lower triangular form of all pairs (p,q) where p != q	# No repeats
	all_keys = {key : None for key in data["distance"].keys()}		# Static all entity keys
	curr_keys = {key : None for key in data["distance"].keys()}		# Place holder entity keys
	total_ents = len(curr_keys.keys())
	print "Total Entities", total_ents
	distance = {}
	for key1 in data["distance"]:	# Full pair p,q lower triangular integer distance matrix enumeration
		curr_keys.pop(key1)
		distance[key1] = {key2 : int(data["distance"][key1][key2]) for key2 in curr_keys.keys()}

	# Loop over min distance D from 0:maxD and find candidate pairs with distance < i
	out_stat1 = {"cluster_num":[],"SumLowerA":[],"SumUpperA":[],"PercentCovered":[]}
	out_stat2 = [{"cluster_num":[],"SumLowerA":[],"SumUpperA":[],"PercentCovered":[]} for p in range(len(max_clusters))]
	for i in range(0,maxD):
		ct2 = 0
		cluster_count = 0
		cluster_list = []
		clusters = {}
		first_cluster = {}
		# Find entity pairs that have distance < i
		#print distance[key1][]
		candidates = {key1:[key2 for key2 in distance[key1].keys() if distance[key1][key2] <= i ] for key1 in all_keys}
		print "# Candidate Pairs",i,len(list(itertools.chain(*[candidates[g] for g in candidates.keys()]))) #,max(candidates),min(candidates),npy.mean(candidates)
		# Determine for all pairs if pairs are to be assigned to new clusters or previous clusters
		#superset[key1] = {key2 : list(itertools.chain(*[self.T_attrs[g] for g in obj1[key2]]))
		#						  for key2 in Tcurr if mem_over[key1][key2]>1}
		for k,keyname in enumerate(candidates.keys()):
			#print i,k,keyname,cluster_count,len(cluster_list)
			for l,keyname2 in enumerate(candidates[keyname]):
				#cluster_list = {key : clusters[i][key] for key in clusters[i]}
				#cluster_list = list(itertools.chain(*[clusters[g] for g in clusters.keys()]))
				if (keyname in cluster_list) and (keyname2 in cluster_list):	# Assign each entity to other's first cluster
					#if first_cluster[keyname] != first_cluster[keyname2]:
					if keyname not in clusters[first_cluster[keyname2]]:
						clusters[first_cluster[keyname2]].append(keyname)
					if keyname2 not in clusters[first_cluster[keyname]]:
						clusters[first_cluster[keyname]].append(keyname2)
						ct2 += 1
						#print "intersecting clusters",ct2,first_cluster[keyname],first_cluster[keyname2],cluster_count
				elif (keyname in cluster_list) and (keyname2 not in cluster_list):	# Assign entity 2 to entity 1's first cluster
					clusters[first_cluster[keyname]].append(keyname2)
					cluster_list.append(keyname2)
					first_cluster[keyname2] = first_cluster[keyname]
				elif keyname2 in cluster_list and (keyname not in cluster_list):	# Assign entity 1 to entity 2's first cluster
					clusters[first_cluster[keyname2]].append(keyname)
					cluster_list.append(keyname)
					first_cluster[keyname] = first_cluster[keyname2]
				else:														# Assign both entities to new cluster list
					clusters[cluster_count] = [keyname,keyname2]
					cluster_list.append(keyname)
					cluster_list.append(keyname2)
					first_cluster[keyname] = cluster_count					# Keep track of current cluster for each key
					first_cluster[keyname2] = cluster_count					# Keep track of current cluster for each key
					cluster_count += 1

		# Determine upper and lower approximations of clusters for total clusters and pruned clusters
		print "Number of Clusters for maxD: ",i," : ",cluster_count
		sum_all = len(list(itertools.chain(*[clusters[g] for g in clusters.keys() if clusters])))
		sum_lower = 0
		sum_upper = 0
		intersections = {}
		int_tmp = {}
		if len(clusters.keys()) > 1:
			for key1 in clusters:
				intersections[key1] = {key2 : list(set(clusters[key1]).intersection(set(clusters[key2])))
								 for key2 in clusters if key2 != key1}
				#print list(itertools.chain(*[intersections[key1][g] for g in intersections[key1]]))
				int_tmp[key1] = len(clusters[key1]) - len(Counter(list(itertools.chain(*[intersections[key1][g] for g in intersections[key1]]))))
				#print intersections[key1]
				#int_tmp = npy.sum([intersections[key1][g] for g in intersections[key1]])
				#print "total, intersections, lower",key1,len(clusters[key1]),int_tmp
				sum_lower += int_tmp[key1] #intersections[key1])
				sum_upper += len(clusters[key1])
		else:
			sum_lower = sum_all
			sum_upper = sum_all

if __name__ == "__main__":

	file2 = open("german_all.json","r")
	data = json.load(file2)
	print data.keys()
	# Numerical encoding for input payload
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

	# Compute distance matrix for all pairs
	#data_length = len(data2[header[0]])
	#print "Data Length",data_length
	#data["distance"] = {}
	#for k in range(0,data_length):
	#	data["distance"][str(k)] = {str(j) : sum([abs(data2[val][k]-data2[val][j]) for val in header]) for j in range(0,data_length)}

	unit = False

	if unit is True:
		data = {"data_set":{}}
		header = ["temp","headache"]
		data["data_set"]["temp"] = [40.8, 40.3, 37.1]
		data["data_set"]["headache"] = [0.3, 0.4, 0.1]
		answer = [True, False, False]
		data["distance"] = {}
		for i in range(3):
			data["distance"][str(i)] = {str(j) : sum([abs(data["data_set"][val][i]-data["data_set"][val][j]) for val in header]) for j in range(0,3)}
		print data["distance"]
		maxD = 2
	else:
		maxD = 18	# Maximum distance for stopping further clustering, if maxD == number of instances, all identities are in all clusters
		max_clusters = [2,3,5,10]	# Used for rejection of clusters when maximizing the sum of lower approximations for all clusters

	# Form lower triangular form of all pairs (p,q) where p != q	# No repeats
	all_keys = {key : None for key in data["distance"].keys()}		# Static all entity keys
	curr_keys = {key : None for key in data["distance"].keys()}		# Place holder entity keys
	total_ents = len(curr_keys.keys())
	print "Total Entities", total_ents
	distance = {}
	for key1 in data["distance"]:	# Full pair p,q lower triangular integer distance matrix enumeration
		curr_keys.pop(key1)
		distance[key1] = {key2 : int(data["distance"][key1][key2]) for key2 in curr_keys.keys()}

	# Loop over min distance D from 0:maxD and find candidate pairs with distance < i
	out_stat1 = {"cluster_num":[],"SumLowerA":[],"SumUpperA":[],"PercentCovered":[]}
	out_stat2 = [{"cluster_num":[],"SumLowerA":[],"SumUpperA":[],"PercentCovered":[]} for p in range(len(max_clusters))]
	for i in range(0,maxD):
		ct2 = 0
		cluster_count = 0
		cluster_list = []
		clusters = {}
		first_cluster = {}
		# Find entity pairs that have distance < i
		#print distance[key1][]
		candidates = {key1:[key2 for key2 in distance[key1].keys() if distance[key1][key2] <= i ] for key1 in all_keys}
		print "# Candidate Pairs",i,len(list(itertools.chain(*[candidates[g] for g in candidates.keys()]))) #,max(candidates),min(candidates),npy.mean(candidates)
		# Determine for all pairs if pairs are to be assigned to new clusters or previous clusters
		#superset[key1] = {key2 : list(itertools.chain(*[self.T_attrs[g] for g in obj1[key2]]))
		#						  for key2 in Tcurr if mem_over[key1][key2]>1}
		for k,keyname in enumerate(candidates.keys()):
			#print i,k,keyname,cluster_count,len(cluster_list)
			for l,keyname2 in enumerate(candidates[keyname]):
				#cluster_list = {key : clusters[i][key] for key in clusters[i]}
				#cluster_list = list(itertools.chain(*[clusters[g] for g in clusters.keys()]))
				if (keyname in cluster_list) and (keyname2 in cluster_list):	# Assign each entity to other's first cluster
					#if first_cluster[keyname] != first_cluster[keyname2]:
					if keyname not in clusters[first_cluster[keyname2]]:
						clusters[first_cluster[keyname2]].append(keyname)
					if keyname2 not in clusters[first_cluster[keyname]]:
						clusters[first_cluster[keyname]].append(keyname2)
						ct2 += 1
						#print "intersecting clusters",ct2,first_cluster[keyname],first_cluster[keyname2],cluster_count
				elif (keyname in cluster_list) and (keyname2 not in cluster_list):	# Assign entity 2 to entity 1's first cluster
					clusters[first_cluster[keyname]].append(keyname2)
					cluster_list.append(keyname2)
					first_cluster[keyname2] = first_cluster[keyname]
				elif keyname2 in cluster_list and (keyname not in cluster_list):	# Assign entity 1 to entity 2's first cluster
					clusters[first_cluster[keyname2]].append(keyname)
					cluster_list.append(keyname)
					first_cluster[keyname] = first_cluster[keyname2]
				else:														# Assign both entities to new cluster list
					clusters[cluster_count] = [keyname,keyname2]
					cluster_list.append(keyname)
					cluster_list.append(keyname2)
					first_cluster[keyname] = cluster_count					# Keep track of current cluster for each key
					first_cluster[keyname2] = cluster_count					# Keep track of current cluster for each key
					cluster_count += 1

		# Determine upper and lower approximations of clusters for total clusters and pruned clusters
		print "Number of Clusters for maxD: ",i," : ",cluster_count
		sum_all = len(list(itertools.chain(*[clusters[g] for g in clusters.keys() if clusters])))
		sum_lower = 0
		sum_upper = 0
		intersections = {}
		int_tmp = {}
		if len(clusters.keys()) > 1:
			for key1 in clusters:
				intersections[key1] = {key2 : list(set(clusters[key1]).intersection(set(clusters[key2])))
								 for key2 in clusters if key2 != key1}
				#print list(itertools.chain(*[intersections[key1][g] for g in intersections[key1]]))
				int_tmp[key1] = len(clusters[key1]) - len(Counter(list(itertools.chain(*[intersections[key1][g] for g in intersections[key1]]))))
				#print intersections[key1]
				#int_tmp = npy.sum([intersections[key1][g] for g in intersections[key1]])
				#print "total, intersections, lower",key1,len(clusters[key1]),int_tmp
				sum_lower += int_tmp[key1] #intersections[key1])
				sum_upper += len(clusters[key1])
		else:
			sum_lower = sum_all
			sum_upper = sum_all

		print "Sum of Lower Approximation for All Clusters",sum_lower
		print "Sum of Upper Approximations for All Clusters",sum_upper
		print "Number of Entities Covered for All Clusters",len(Counter(cluster_list).keys())
		print "Percentage of Entities Covered for All Clusters",(len(Counter(cluster_list).keys())/float(total_ents))*100.0
		out_stat1["cluster_num"].append(cluster_count)
		out_stat1["SumLowerA"].append(sum_lower)
		out_stat1["SumUpperA"].append(sum_upper)
		out_stat1["PercentCovered"].append((len(Counter(cluster_list).keys())/float(total_ents))*100.0)

		# Prune the clusters to maximize upper approximations
		cluster_upper_approx = {g : len(clusters[g]) for g in clusters}
		tmpmem = sorted(cluster_upper_approx.iteritems(), key=operator.itemgetter(1),reverse=True)
		#print "1",tmpmem
		#tmpmem = sorted(int_tmp.iteritems(), key=operator.itemgetter(1),reverse=True)
		#print "2",tmpmem
		clusters1 = []
		cluster_count1 = []
		cluster_list1 = []
		for p,value in enumerate(max_clusters):
			sorted_clusters = [t[0] for t in tmpmem[0:max_clusters[p]]]
			clusters1.append({key : clusters[key] for key in sorted_clusters})
			cluster_count1.append(len(clusters1[p].keys()))
			cluster_list1.append(list(itertools.chain(*[clusters1[p][g] for g in clusters1[p].keys()])))
			#print "Pruned Clusters for maxD: ",i," and maxClusters: ",max_clusters[p]," : ",cluster_count1[p],len(cluster_list1[p])

			# Compute upper/lower approximations for pruned clusters
			sum_all_1 = len(list(itertools.chain(*[clusters1[p][g] for g in clusters1[p].keys() if clusters1])))
			sum_lower1 = 0
			sum_upper1 = 0
			intersections1 = {}
			if len(clusters1[p].keys()) > 1:
				for key1 in clusters1[p]:
					#print key1
					intersections1[key1] = {key2 : list(set(clusters1[p][key1]).intersection(set(clusters1[p][key2])))
									 for key2 in clusters1[p] if key2 != key1}
					#print list(itertools.chain(*[intersections[key1][g] for g in intersections[key1]]))
					int_tmp1 = len(Counter(list(itertools.chain(*[intersections1[key1][g] for g in intersections1[key1]]))))
					#int_tmp = npy.sum([intersections[key1][g] for g in intersections[key1]])
					#print "total, intersections, lower",key1,len(clusters[key1]),int_tmp
					sum_lower1 += (len(clusters1[p][key1]) - int_tmp1) #intersections[key1])
					sum_upper1 += len(clusters1[p][key1])
					#print len(clusters1[p][key1]),int_tmp1,sum_upper1,sum_lower1

			else:
				sum_lower1 = sum_all_1
				sum_upper1 = sum_all_1
			print "Results for : ",max_clusters[p]," Pruned Clusters"
			print "Sum of Lower Approximation for Pruned Clusters :",sum_lower1
			print "Sum of Upper Approximations for Pruned Clusters",sum_upper1
			print "Number of Entities Covered for Pruned Clusters",len(Counter(cluster_list1[p]).keys())
			print "Percentage of Entities Covered for Pruned Clusters",(len(Counter(cluster_list1[p]).keys())/float(total_ents))*100.0

			# Pack stats into output and plot
			out_stat2[p]["cluster_num"].append(cluster_count1[p])
			out_stat2[p]["SumLowerA"].append(sum_lower1)
			out_stat2[p]["SumUpperA"].append(sum_upper1)
			out_stat2[p]["PercentCovered"].append((len(Counter(cluster_list1[p]).keys())/float(total_ents))*100.0)


# TODO determine optimal maxD to report results

	# Print stats for members of clusters
	# Determine labels
	list1 = [i for i in range(len(data["response"])) if data["response"][i] == '1']
	list2 = [i for i in range(len(data["response"])) if data["response"][i] == '2']
	print list1

	tableau_lists = []
	tableau_1 = []
	tableau_2 = []
	for key in header:
		tableau_lists.append(data2[key][:])
		tableau_1.append(npy.asarray(data2[key])[list1].tolist())
		tableau_2.append(npy.asarray(data2[key])[list2].tolist())
	datav = npy.asarray(tableau_lists).T
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
	listhash = {"data_set_map":[]}
	lrnr = clustering.clustering(GlowfishException,listhash)
	output = lrnr.clusterLearn(data["data_set"],2)
	print output.keys()
	meank = [[] for g in range(len(output["group_names"]))]
	val = [[] for n in range(len(output["group_names"]))]
	for m in range(len(output["group_predictions"])):
		for n in range(len(output["group_names"])):
			if output["group_predictions"][m] == n:
				val[n].append(data["response"][m])
				meank[n].append(datav[int(val[n][-1]),:])
	meankp = []
	stddevk = []
	for n in range(len(output["group_names"])):
		print Counter(val[n])
		meankp.append(npy.mean(meank[n],axis=0))
		stddevk.append(npy.std(meank[n],axis=0))

	from sklearn.decomposition import PCA
	pca = PCA(n_components=2)
	X1_r = pca.fit(data1)
	loadings = X1_r.components_
	#print "Good",loadings
	plt.scatter(*loadings, alpha=0.3, c="r",label="Good")
	X2_r = pca.fit(data3)
	loadings = X2_r.components_
	#print "Bad",loadings
	plt.hold(True)
	plt.scatter(*loadings, alpha=0.3, c="b",label="Bad")
	plt.title("First 2 Components of PCA Loadings for Credit Risk")
	plt.show()


	response_all = {"response":data["response"]}
	listhash2 = {'data_set_map':{}}
	for key in data2:

		if any(isinstance(n,float) for n in data2[key][:]):
			value = float
		elif any(isinstance(n,int) for n in data2[key][:]):
			value = int
		elif any(isinstance(n, (str, unicode)) for n in data2[key][:]):
			value = str
		listhash2['data_set_map'][key] = value

	# Run ANOVA and produce significance results
	anova = featureSelection.featureSelection(GlowfishException,listhash)
	output = anova.featureSelect(data2,response_all)
	output["sig"] = -npy.log10(output["significance_factors"])
	print "ANOVA and Ranking Results:"
	for g in range(0,len(output["significance_factors"])):
		print output["feature_names"][g],output["sig"][g]

	resultsm = []
	resultss = []
	rangek = [l+.2 for l in range(20)]
	ranger = [l+.1 for l in range(20)]
	print "total instances",Counter(data["response"])
	for key1 in range(len(clusters1)):
		print key1
		fig, axs = plt.subplots(nrows=1,ncols=1)
		axs.errorbar(range(20),mean2,fmt='ro',yerr=std2,label="True Good Centroid")
		plt.hold(True)
		axs.errorbar(range(20),mean1,fmt='bo',yerr=std1,label="True Bad Centroid")
		axs.errorbar(rangek,meankp[1],fmt='r+',yerr=stddevk[0],label="Kmeans 0")
		axs.errorbar(rangek,meankp[0],fmt='b+',yerr=stddevk[1],label="Kmeans 1")

		for key in clusters1[key1]:
			datav2 = []
			meant = []
			stdt = []
			#print key,npy.mean(npy.asfarray(clusters1[0][key])),npy.std(npy.asfarray(clusters1[0][key]))
			for val in clusters1[key1][key]:
				meant.append(data["response"][int(val)])
				datav2.append(datav[int(val),:])
			#print meant
			tmp = npy.mean(npy.asarray(datav2),axis=0)
			tmp2 = npy.std(npy.asarray(datav2),axis=0)
			if key1 == 1:
				if key == 4:
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

	if False:
		for q in range(len(max_clusters)):
			plt.subplot(3,1,1)
			plt.title("Full Cluster Stats verus "+str(max_clusters[q])+" Pruned Clusters")
			plt.plot(out_stat1["cluster_num"],'r',label="All Clusters")
			plt.hold(True)
			plt.plot(out_stat2[q]["cluster_num"],'ro',label="Partial Clusters")
			plt.plot(out_stat1["PercentCovered"],'b',label="Full - % Data Covered")
			plt.plot(out_stat2[q]["PercentCovered"],'bo',label="Partial - % Data Covered")
			plt.legend()
			plt.subplot(3,1,2)
			plt.title("Full Cluster UpperA/LowerA verus "+str(max_clusters[q])+" Pruned Clusters")
			plt.plot(out_stat1["SumLowerA"],'r',label="Full Sum(LowerA)")
			plt.hold(True)
			plt.plot(out_stat2[q]["SumLowerA"],'ro',label="Pruned Sum(LowerA)")
			plt.plot(out_stat1["SumUpperA"],'b',label="Full Sum(UpperA)")
			plt.plot(out_stat2[q]["SumUpperA"],'bo',label="Pruned Sum(UpperA)")
			plt.legend()
			plt.xlabel("Max Distance Measure Used",fontsize=16)
			plt.subplot(3,1,3)
			plt.title("Cluster Centroids"+str(max_clusters[q])+" Pruned Clusters")
			plt.plot(resultsm[q],"ro")
			plt.hold(True)
			plt.plot(resultss[q],'bo')
			plt.show()

	file2 = open("results_mean.txt","w")
	for name in header:
		file2.writelines(name+" ")
	file2.writelines("\n")
	for n in range(len(header)):
		file2.writelines(str(meankp[0][n])+" ")
		file2.writelines(str(stddevk[0][n])+" ")
		file2.writelines(str(meankp[1][n])+" ")
		file2.writelines(str(stddevk[1][n])+" ")
		file2.writelines(str(resultsm[0][n])+" ")
		file2.writelines(str(resultss[0][n])+" ")
		file2.writelines(str(resultsm[1][n])+" ")
		file2.writelines(str(resultss[1][n])+" ")
		file2.writelines(str(resultsm[2][n])+" ")
		file2.writelines(str(resultss[2][n])+" ")
		file2.writelines(str(resultsm[3][n])+" ")
		file2.writelines(str(resultss[3][n])+" ")
		file2.writelines(str(resultsm[4][n])+" ")
		file2.writelines(str(resultss[4][n])+" ")
		file2.writelines(str(mean1[n])+" ")
		file2.writelines(str(std1[n])+" ")
		file2.writelines(str(mean2[n])+" ")
		file2.writelines(str(std2[n])+" ")
		file2.writelines("\n")
	file2.close()