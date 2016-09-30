#!/usr/bin/env python2.7
# encoding: utf-8

"""
@Desciption

	Streamning Naive Bayes Classifier including dynamic re-binning, training, predictions, and classification
	statistics. Handles heterogeneous feature sets with mixed INT, FLOAT, STRING feature types. Also handles missing
	values from feature set.

	Methods Implemented :
	1. classLearn() - Training using continuous (float), categorical (int or string), and mixed data types
	2. classPredict() - Prediction based on Naive Bayes Classifier
	3. classUpdate() - Update method for incremental training
	4. classAcc() - Accuracy computation method

	Includes dynamic binning to account for updated feature dynamic ranges over data stream

@notes
	re-binning implemented here is quite crude

	need to optimize how I bin for both int and float attributes given that for streaming I need to have same bins
	for all subsequent calls to predicter---thus, initial bining must be optimal for all future learning / training

@author Michael Tompkins
@copyright 2016
"""

#Externals
import numpy as npy
import time
from collections import Counter
import copy

class classifier:


	def __init__(self, listmap):

		"""
		Class for Learning, Predicting, Updating, and Accuracy Check of Naive Bayes Classifier
		"""

		self.listmap = listmap['data_set_map']
		self.debug = True
		self.small = 1.0e-10


	def classUpdate(self,freqs,params,update,response):

		"""
		Naive Bayesian Classifier Update Method
		Incrementally augmnent priors and frequency tables (and potentialy binning) with
		attributes of [update] set from previous training

		:arg freqs: dictionary of frequency tables for training including priors/likelihood estimations
		:arg params: dictionary of states for previous feature space stream
		:arg update:
		:arg response: dictionary of current target classes for current portion of feature space stream
		"""

		t1 = time.time()
		int_range = params['int_range']
		bin_inc = params['bin_inc']
		int_bins = params['int_bins']
		int_min = params['int_min']
		int_max = params['int_max']
		float_range = params['float_range']
		float_inc = params['float_inc']
		float_bins = params['float_bins']
		float_min = params['float_min']
		float_max = params['float_max']
		numclass = params['numclass'] # Check
		classes = params['classes'] # Check
		strings = params['strings'] # Check

		# Check feature label integrity between train and predict
		if len(update.keys()) == len(freqs[0].keys()):
			for key1 in update:
				if key1 not in freqs[0].keys():
					raise Exception("Feature label mismatch. Specify new set if needed.")
		else:
			raise Exception("Feature label mismatch. Set reset_model to true or specify new set if needed.")

		try:

			length = len(update[update.keys()[0]])
			keyval = response.keys()
			if len(keyval) > 1:
				raise Exception("More than 1 key for response data dictionary is not allowed")

			# Add new classes as they appear in update
			classlstnew = response[keyval[0]]
			newclasses = list(npy.unique(classlstnew))

			classesk = copy.copy(classes)			# Subset of known classes for updates
			indxk = []
			for m,namek in enumerate(classesk):
				indxk.append([i for i in range(0,length) if response[keyval[0]][i] == namek])

			# Handle case when new class(es) are encountered for first time
			classesuk = []
			for name in newclasses:
				if name not in classesk:			# Update classes
					classesuk.append(name)			# Unknown classes
					classes.append(name)			# Local scope update
					params['classes'] = classes		# Output update
					numclass += 1					# Local scope update
					params['numclass'] += 1			# Output update
					params['class_ct'].append(0)
					#params['class_ct'].append(Counter(classlstnew)[name])
					indxk.append([i for i in range(0,length) if response[keyval[0]][i] == name])
					freqs.append({k:[freqs[0][k][q]*0 for q in range(0,len(freqs[0][k]))] for k in update.keys()})

			for j,classname in enumerate(classes): # Loop over each known class
				params['class_ct'][j] = params['class_ct'][j] + (Counter(classlstnew)[classname])
				for i in indxk[j]:  					# loop over each row in [update] for class j
					for key in update:  				# Loop over number of attributes per row
						if isinstance(update[key][i],int):	# If value is INT and not None
							if (update[key][i] > bin_inc[key][-1]+int_bins[key][-1]) or \
									(update[key][i] < int_bins[key][0]):
								if update[key][i] > bin_inc[key][-1]+int_bins[key][-1]:				# Add bin to left
									newinc = (update[key][i]-int_max[key])
									newbin = update[key][i]
									bin_inc[key].append(newinc)
									params['bin_inc'][key] = bin_inc[key]
									int_bins[key].append(newbin)
									params['int_bins'][key] = int_bins[key]
									int_range[key] = abs(int_min[key] - update[key][i])
									params['int_range'][key] = int_range[key]
									int_max[key] = update[key][i]
									params['int_max'][key] = int_max[key]
									for p,nam in enumerate(classes):
										freqs[p][key].append(0)
									freqs[j][key][-1] = 1
								else:																# Add bin to right
									newinc = (int_min[key] - update[key][i])
									newbin = update[key][i]
									bin_inc[key].insert(0,newinc)
									params['bin_inc'][key] = bin_inc[key]
									int_bins[key].insert(0,newbin)
									params['int_bins'][key] = int_bins[key]
									int_range[key] = abs(int_max[key] - update[key][i])
									params['int_range'][key] = int_range[key]
									int_min[key] = update[key][i]
									params['int_max'][key] = int_min[key]
									for p,nam in enumerate(classes):
										freqs[p][key].insert(0,0)
									freqs[j][key][0] = 1
							else:
								ct = 0
								for k in range(0,len(bin_inc[key])):  # Loop over range in int range
									val = int_bins[key][k]
									if val <= update[key][i] <= (val + bin_inc[key][k]):
										freqs[j][key][ct] += 1
									ct += 1
								int_range[key] = max([abs(params['int_max'][key]-update[key][i]),
												 abs(update[key][i]-params['int_min'][key]),int_range[key]])
								params['int_min'][key] = min([update[key][i],params['int_min'][key]])
								int_min[key] = params['int_min'][key]
								params['int_max'][key] = max([update[key][i],params['int_max'][key]])
								int_max[key] = params['int_max'][key]
								params['int_range'][key] = int_range[key]

						elif isinstance(update[key][i],float) and not npy.isnan(update[key][i]):
							if (update[key][i] > (float_max[key]+self.small)) or (update[key][i] < (float_min[key]-self.small)):
								# Add a bin to appropriate side
								if update[key][i] > float_max[key]:			# Add bin to left
									newinc = (update[key][i]-float_max[key])
									newbin = update[key][i]
									float_inc[key].append(newinc)
									params['float_inc'][key] = float_inc[key]
									float_bins[key].append(newbin)
									params['float_bins'][key] = float_bins[key]
									float_range[key] = abs(float_min[key] - update[key][i])
									params['float_range'][key] = float_range[key]
									float_max[key] = update[key][i]
									params['float_max'][key] = update[key][i]
									for p,nam in enumerate(classes):
										freqs[p][key].append(0)
									freqs[j][key][-1] = 1
								else:																# Add bin to right
									newinc = (float_min[key] - update[key][i])
									newbin = update[key][i]
									float_inc[key].insert(0,newinc)
									params['float_inc'][key] = float_inc[key]
									float_bins[key].insert(0,newbin)
									params['float_bins'][key] = float_bins[key]
									float_range[key] = abs(float_max[key] - update[key][i])
									params['float_range'][key] = float_range[key]
									float_min[key] = update[key][i]
									params['float_min'][key] = update[key][i]
									for p,nam in enumerate(classes):
										freqs[p][key].insert(0,0)
									freqs[j][key][0] = 1
							else:
								for k in range(0,len(float_inc[key])):  # Loop over range in int range of attribute values
									val = float_bins[key][k]
									if val <= update[key][i] <= val + float_inc[key][k]:  # Attribute value = k
										freqs[j][key][k] += 1
										float_range[key] = max([abs(params['float_max'][key]-update[key][i]),
														 abs(update[key][i]-params['float_min'][key]),float_range[key]])
										float_min[key] = min([update[key][i],params['float_min'][key]])
										params['float_min'][key] = float_min[key]
										float_max[key] = max([update[key][i],params['float_max'][key]])
										params['float_max'][key] = float_max[key]
										params['float_range'][key] = float_range[key]

						if isinstance("str",self.listmap[key]):
							if update[key][i] is None:
								pass
							else:
								if update[key][i] not in strings[key]:	# Add string to table as needed
									params['strings'][key].append(update[key][i])	# Output update
									strings[key] = params['strings'][key]	# Local scope update
									for p,nam in enumerate(classes):
										freqs[p][key].append(0)
									freqs[j][key][-1] = 1
								else:
									for k,name in enumerate(strings[key]):
										if update[key][i] == name:
											freqs[j][key][k] += 1

			params["version"] = str(time.time())

			t2 = time.time()
			if self.debug is True:
				print "classUpdate Took: "+str(t2-t1)

		except Exception as e:
			raise Exception(e)

		return freqs,params


	def classLearn(self,train,response):

		"""
		Naive Bayesian Classifier Initial Training Method - First call to classifier class
		Train and build frequency table from initial training feature set stream
		"""

		try:
			t1 = time.time()

			# Row Count and Sanity Check for Bayes Classifier
			rowtmp = []
			for key in train:
				rowtmp.append(len(train[key]))  # Row count of each attribute for the training set

			#if min(rowtmp) != max(rowtmp):  # If row counts are not equal for all attributes, except
			#    raise self.exceptionClass(
			#        "422.2")  # Unprocessable Entity/Request in Learner Method - Row count for all keys not equal
			rowcount = rowtmp[0]  # If all keys have equal rows, then assign row count to 1st index

			# Attribute Count for Classifier
			numatt = len(train)  # Number of attributes

			# Determine number of unique classes from target dictionary
			keyval = response.keys()
			if len(keyval) > 1:
				raise Exception ("more than 1 key for target data")

			classlist = []
			for row in response[keyval[0]]:
				classlist.append(row)

			classes = list(npy.unique(classlist))
			numclass = len(classes)

			#Convert to ints from numpy int64 as needed
			if isinstance(classes[0],int):
				classes2 = []
				for g in classes:
					classes2.append(int(g))
				classes = classes2

			class_ct = [0 for i in range(0,numclass)]
			for row in response[keyval[0]]:
				for j,classname in enumerate(classes):
					if classname == row:
						class_ct[j] += 1

			if self.debug:
				print class_ct
				print 'rowcount, numatt, numclass'
				print rowcount,numatt,numclass
				print classes
				print numclass

			# Determine number of unique strings in each dictionary attributes
			stringlist = {}
			strings = {}
			str_range = {}
			for key in train:
				#if isinstance(train[key][0], str) or isinstance(train[key][0], unicode):
				# if self.listmap[key] is str or self.listmap[key] is unicode:
				if isinstance("str",self.listmap[key]):
					stringlist[key] = []
					stringlist[key].append(train[key][0])
					for entry in train[key][1:]:
						stringlist[key].append(entry)

			for key in stringlist:
				strings[key] = list(npy.unique(stringlist[key]))
				str_range[key] = len(strings[key])

			if self.debug:
				t2 = time.time()
				print t2 - t1

			# Determine range bin size for float and ints for each attribute
			int_range = {}
			bin_inc = {}
			float_range = {}
			float_inc = {}
			int_bins = {}
			float_bins = {}
			int_min = {}
			int_max = {}
			float_min = {}
			float_max = {}
			bin_min = {}

			freqdict = []
			indexes = []
			for m in range(0,numclass):
				indexes.append([])
				classname = classes[m]
				freqdict.append({})
				indexes[m] = [i for i,x in enumerate(classlist) if x == classname]

			# Build frequency table for int/float attributes
			for key in train:
				if self.listmap[key] is int:
					tmplist = [i for i in train[key][:] if isinstance(i,int)]
					bin_min[key] = len(npy.unique(tmplist))
					int_range[key] = npy.max(tmplist) - npy.min(tmplist)
					int_min[key] = npy.min(tmplist)
					int_max[key] = npy.max(tmplist)
					tmpl = int_range[key] / 6
					bin_num = npy.max([tmpl,1])
					bin_tmp = int_range[key] / bin_num + 1
					bin_inc[key] = [bin_num + 1 for i in range(0,6)]
					int_bins[key] = [int_min[key]]

					for g in range(0,len(bin_inc[key])):
						int_bins[key].append(int_bins[key][g] + bin_inc[key][g])

					for j,classname in enumerate(classes):  # Enumerate over classes
						T = [train[key][i] for i in indexes[j] if isinstance(train[key][i],int)]
						if j == 0:
							bins = int_bins[key]
							[values,bins] = npy.histogram(T,bins,(int_min[key],int_max[key]))
						else:
							[values,bins] = npy.histogram(T,bins,(int_min[key],int_max[key]))

						freqdict[j][key] = list(values)

				elif self.listmap[key] is float:
					tmplist = [i for i in train[key][:] if not npy.isnan(i)] #isinstance(i,float)]
					if len(tmplist) == 0:
						float_range[key] = 0.0
						float_min[key] = 0.0
						float_max[key] = 0.0
						float_bins[key] = [0.0]
						float_inc[key] = [0.0]

					else:
						float_range[key] = npy.ceil(npy.max(tmplist) - npy.min(tmplist))
						float_min[key] = npy.min(tmplist)
						float_max[key] = npy.max(tmplist)
						[values,binsf] = npy.histogram(tmplist,6)
						float_bins[key] = list(binsf)
						float_inc[key] = []
						for g in range(0,len(float_bins[key]) - 1):
							float_inc[key].append(float_bins[key][g + 1] - float_bins[key][g])

					for j,classname in enumerate(classes):  # Enumerate over classes
						T = [train[key][i] for i in indexes[j] if isinstance(train[key][i],float)]
						if j == 0:
							bins = float_bins[key]
							[values,bins] = npy.histogram(T,bins,(float_min[key],float_max[key]))
						else:
							[values,bins] = npy.histogram(T,bins,(float_min[key],float_max[key]))

						freqdict[j][key] = list(values)

				elif isinstance("str",self.listmap[key]):
					for j,classname in enumerate(classes):  # Enumerate over classes
						freqdict[j][key] = [0 for n in range(0,str_range[key])]
						T = [train[key][i] for i in indexes[j]]
						str_cntr = Counter(T)
						for k,stringkey in enumerate(strings[key]):
							freqdict[j][key][k] = str_cntr[stringkey]

				else:
					raise Exception("Unprocessable Feature Type (int,float,string supported)")

			t2 = time.time()

			if self.debug:
				print 'classLearn - Timing'+str(t2-t1)
				#print 'bin_min',bin_min
				#print 'int_bins',int_bins
				#print 'float_bins',float_bins
				#print 'bin_inc',bin_inc
				#print 'int_range',int_range
				#print 'int_min',int_min
				#print 'float_inc',float_inc
				#print 'float_min',float_min
				#print 'float_range',float_range

				#print "All Frequency Tables Took:"
				t3 = time.time()
				print t3 - t2

			if self.debug:
				t4 = time.time()
				print t4 - t3

		except Exception as e:
			raise Exception(e)

		paramdict = {'int_range': int_range,'bin_inc': bin_inc,'int_bins': int_bins,'int_min': int_min,
					 'float_range': float_range,'float_inc': float_inc,
					 'float_bins': float_bins,'float_min': float_min,
					 'numatt': numatt,'numclass': numclass,'classes': classes,'class_ct': class_ct,
					 'strings': strings, 'int_max': int_max, 'float_max': float_max}

		if self.debug:
			print paramdict
			print len(freqdict)
			for j in range(0,len(classes)):
				print j,freqdict[j]

		return freqdict,paramdict


	def classPredict(self,freqs,params,test):

		"""
		Naive Bayes Predicter method
		Build posteriors from attributes of [test] set and priors from training
		"""

		int_range = params['int_range']
		bin_inc = params['bin_inc']
		int_bins = params['int_bins']
		int_min = params['int_min']
		float_range = params['float_range']
		float_inc = params['float_inc']
		float_bins = params['float_bins']
		float_min = params['float_min']
		numatt = params['numatt']
		numclass = params['numclass']
		classes = params['classes']
		num = params['class_ct']
		numtot = npy.sum(num)
		strings = params['strings']

		# Check feature label integrity between train and predict
		if len(test.keys()) == len(freqs[0].keys()):
			for key1 in test:
				if key1 not in freqs[0].keys():
					raise Exception("Feature labels mismatch between train and predict classification data")
		else:
			raise Exception("Wrong length feature set" + str(len(test.keys())) + " : " + str(len(freqs[0].keys())))

		try:
			length = len(test[test.keys()[0]])
			ksm = 1
			m = 2
			classtmp = None
			probtmp = None
			exptmp = None
			class_dict = {'classes': classes,'class_predictions': [],'prob_for_predictions': [],
						  'exp_for_predictions': []}
			for i in range(0,length):  # loop over each row in [test]
				prob = -1000000  # Initial likelihood of each class (0,1)
				for j,classname in enumerate(classes):  # Loop over hypotheses (each of the classes)
					prior = (num[j] + ksm) / float((numtot + (ksm * numclass)))
					log_p = npy.log(prior)
					for key in test:  # Loop over number of attributes
						if self.listmap[key] is int:  # If int, enumerate
							ct = 0
							for k in range(0,len(bin_inc[key])):  # Loop over range in int range
								val = int_bins[key][k]
								if val <= test[key][i] <= (val + bin_inc[key][k]):
									inc = (freqs[j][key][ct] + (m * prior)) / float((num[j] + m))
									log_p += npy.log(inc)
								ct += 1

						elif self.listmap[key] is float:
							for k in range(0,len(float_inc[key])):  # Loop over range in int range of attribute values
								val = float_bins[key][k]
								if val <= test[key][i] <= val + float_inc[key][k]:  # Attribute value = k
									inc = (freqs[j][key][k] + (m * prior)) / float((num[j] + m))
									log_p += npy.log(inc)

						elif isinstance("str",self.listmap[key]):
							for k,name in enumerate(strings[key]):
								if test[key][i] == name:
									inc = (freqs[j][key][k] + (m * prior)) / float((num[j] + m))
									log_p += npy.log(inc)

					if log_p >= prob:
						prob = copy.copy(log_p)
						classtmp = copy.copy(classname)
						probtmp = copy.copy(prob)
						try:
							exptmp = npy.exp(prob)
						except:
							exptmp = None

				class_dict['class_predictions'].append(classtmp)
				class_dict['prob_for_predictions'].append(round(probtmp,2))
				class_dict['exp_for_predictions'].append(round(exptmp,3))

		except Exception as e:
			raise Exception(e)

		return class_dict


	def classAcc(self,class_dict,responsedata,tagname="all",metadata=None):

		"""
		Determine precision, recall, F-1 score and composite accuracy for each class predicted
		"""

		#Determine keyname
		keyname = responsedata.keys()[0]

		#First pull appropriate rows for certain tag if metadata exists
		if metadata is not None:
			taglist = [i for i in range(0,len(metadata["tag"])) if tagname in metadata["tag"][i]]
			print len(taglist),len(responsedata[keyname]),len(metadata["tag"])
		else:
			taglist = [i for i in range(0,len(responsedata[keyname]))]

		acc_dict = {}
		crct = [0 for i in range(0,len(class_dict['classes']))]  #Counter for each class
		fpos = [0 for i in range(0,len(class_dict['classes']))]  #Counter for each class
		fneg = [0 for i in range(0,len(class_dict['classes']))]  #Counter for each class
		acc_dict['precision'] = [0 for i in range(0,len(class_dict['classes']))]
		acc_dict['recall'] = [0 for i in range(0,len(class_dict['classes']))]
		acc_dict['f1_scores'] = [0 for i in range(0,len(class_dict['classes']))]
		acc_dict['class_names'] = [0 for i in range(0,len(class_dict['classes']))]

		# Check that type for classes are consistent from train and test sets
		if type(class_dict['classes'][0]) != type(responsedata[keyname][0]):
			raise Exception("Response array types mismatch between train and predict classification data")

		try:
			rows = len(responsedata[keyname][:])
			#for i in range(0, rows):
			for i in taglist:
				for j in range(0,len(class_dict['classes'])):
					if class_dict['class_predictions'][i] == class_dict['classes'][j] and \
									responsedata[keyname][i] == class_dict['classes'][j]:
						crct[j] += 1  # True positives for class j
					elif class_dict['class_predictions'][i] == class_dict['classes'][j]:
						fpos[j] += 1  # False positive for class j
					else:
						fneg[j] += 1  # False negative for class j

			# Compute precision, recall, and F-1 score for each class
			class_cnt = []
			for m,classname in enumerate(class_dict['classes']):
				tmpclasslist = [responsedata[keyname][i] for i in taglist]
				class_cnt.append(Counter(tmpclasslist)[class_dict['classes'][m]])  # Number of outcomes for each class
				# Handle case for crct == 0
				if crct[m] == 0:
					acc_dict['recall'][m] = 0.0
					acc_dict['f1_scores'][m] = 0.0
					acc_dict['precision'][m] = 0.0
					acc_dict['class_names'][m] = classname
				else:
					precision = crct[m] / float(fpos[m] + crct[m])
					recall_cnt = crct[m]  # Number of hits for this class
					acc_dict['recall'][m] += round (recall_cnt / float(class_cnt[m]),2) # Recall for class

					# Compute F-1 Score for each class
					acc_dict['f1_scores'][m] = round( 2 * ((precision * acc_dict['recall'][m]) /
														   (precision + acc_dict['recall'][m]) ) ,2) # F-1 Score

					acc_dict['precision'][m] = round(precision,2)
					acc_dict['class_names'][m] = classname

			# Compute final composite accuracy for all classes from mean of F1 scores across all classes
			rlt = range(0,len(class_dict['classes']))
			acc_dict['Composite_Accuracy'] = round(sum([acc_dict['f1_scores'][m]*class_cnt[m] for m in rlt])/sum(class_cnt),2)

		except Exception as e:
			raise Exception(e)

		return acc_dict


if __name__ == "__main__":

	"""
	Class-Level Tester
	"""

	from sys import argv
	import json

	if argv[1] == "train":  # Training
		try:
			# Load data from file
			data_file = argv[2]
			dfile = open(data_file, "r")
			data1 = json.load(dfile)  # Initial independent variable dataset
			try:
				response_file = argv[3]
				rfile = open(response_file, "r")
				responsedata = json.load(rfile)  # Initial response dataset
			except:
				responsedata = data1['response']
				data1.pop("response")

			try:
				data = data1["data_set"]
			except:
				data = data1

		except Exception as e:
			raise Exception('One or more of your input files are absent for unreadable')

		#Validate data
		listhash = {'data_set_map':{}}
		for key in data:
			#print key
			if any(isinstance(n,float) for n in data[key][:]):
				value = float
			elif any(isinstance(n,int) for n in data[key][:]):
				value = int
			elif any(isinstance(n, (str, unicode)) for n in data[key][:]):
				value = str
			listhash['data_set_map'][key] = value

		#Instantiate learner class
		lrner = classifier(listhash)

	if argv[1] == "test":  # Testing/Predictions
		# Read priors_dict and pars_dict from files for classifications
		# This will be replaced by server cookie or DB storage using userid instead of dummy file names

		try:
			# Load data from file
			data_file = argv[2]
			file1 = open("priors.json", "r")
			priors_dict = json.load(file1)
			file1.close()
			file2 = open("params.json", "r")
			pars_dict = json.load(file2)
			file2.close()
			dfile = open(data_file, "r")
			data1 = json.load(dfile)  # Initial independent variable dataset
			try:
				data = data1["data_set"]
			except:
				data = data1
			#data = data1["data"]

		except Exception as e:
			raise Exception('One or more of your input files are absent for unreadable')

		#Validate data
		listhash = {'data_set_map':{}}
		for key in data:
			if any(isinstance(n,float) for n in data[key][:]):
				value = float
			elif any(isinstance(n,int) for n in data[key][:]):
				value = int
			elif any(isinstance(n, (str, unicode)) for n in data[key][:]):
				value = str
			listhash['data_set_map'][key] = value

		#Instantiate learner class
		lrner = classifier(listhash)

	if argv[1] == "train":  # Training
		#data_dict = lrner.dataNorm(data,responsedata)		# Z-score normalization of data for PC regression
		#[data_dict['covdat'],data_dict['eigs'],data_dict['vals']] =  lrner.pcaCompute(data_dict)	# PCA compression for PC regression
		#regr_model_dict = lrner.pcaLearn(data_dict)
		#print regr_model_dict

		#lrner2 = classifier(GlowfishException,listhash)
		t1= time.time()
		for k in range(0,1): #100):
			#mind = k*800
			mind = 0
			maxd = 1000000
			#maxd = (k+1)*800
			datatmp = {k:data[k][mind:maxd] for k in data.keys()}
			responsetmp = {k:responsedata[k][mind:maxd] for k in responsedata.keys()}
			print len(datatmp[datatmp.keys()[0]][:]),mind,maxd,k
			print len(responsetmp[responsetmp.keys()[0]][:])
			if k == 0:
				[priors_dict, pars_dict] = lrner.classLearn(datatmp, responsetmp)  # Run learner
				#print priors_dict[0]
				#print priors_dict[1]
				#print pars_dict['float_bins']['euribor3m'],pars_dict['float_inc']['euribor3m']
				#print pars_dict['float_bins']['nr.employed'],pars_dict['float_inc']['nr.employed']
				for h in pars_dict['int_bins'].keys():
					print h,len(pars_dict['int_bins'][h]),pars_dict['int_bins'][h],pars_dict['int_max'][h],pars_dict['int_min'][h]
				for h in pars_dict['float_bins'].keys():
					print h,len(pars_dict['float_bins'][h])
				for h in pars_dict['strings'].keys():
					print h,pars_dict['strings'][h]
				#print 'marital',priors_dict[0]["marital"] #,priors_dict[1]["marital"]
			else:
				[priors_dict, pars_dict] = lrner.classUpdate(priors_dict, pars_dict, datatmp, responsetmp)  # Run learner
				# print priors_dict[0]
				# print priors_dict[1]
				for h in pars_dict['int_bins'].keys():
					print h,len(pars_dict['int_bins'][h]),pars_dict['int_bins'][h],pars_dict['int_max'][h],pars_dict['int_min'][h]
				for h in pars_dict['float_bins'].keys():
					print h,len(pars_dict['float_bins'][h])
				for h in pars_dict['strings'].keys():
					print h,pars_dict['strings'][h]
				#print 'marital',priors_dict[0]["marital"] #,priors_dict[1]["marital"]
		# Write priors_dict and pars_dict to files for later classifications
		# This will be replaced by server cookie or DB storage using userid instead of dummy file names
		t2= time.time()
		print 'total_time',t2-t1
		file1 = open("priors.json", "w")
		#print priors_dict
		#json.dumps(priors_dict)
		json.dump(priors_dict, file1)
		file1.close()
		file2 = open("params.json", "w")
		json.dump(pars_dict, file2)
		file2.close()

	else:  # Testing/Predictions
		class_dict2 = lrner.classPredict(priors_dict, pars_dict, data)  # Run predicter

		# Run precision,recall,F1 statistics
		response_file = argv[3]
		rfile = open(response_file, "r")
		responsedata = json.load(rfile)
		print 'done with predict'
		outdict = lrner.classAcc(class_dict2, responsedata,"all")
		print outdict
