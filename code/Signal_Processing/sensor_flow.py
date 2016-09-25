#!/usr/bin/env python2.7
# encoding: utf-8

"""
@additions - New stuff in update version of sensor_flow
	1. Fixed issue with 0 states array when sensor comes back online by back propagating current state for sensor
		to all previous states in moving window
	2. Fixed issue of new_sensor == sensor_list for initial run being referenced before assignment
	3. Made DSP processing window (mov_win) relative to sensor periodicity or max(min_window,2.2xmax(previous_periods))
	4. Made anomaly density window (mem_win) relative to sensor primary periodicity or mem_win_min default value

@todo
ISSUE : when mov_win reduced due to local periodicity change < 500 samples, then larger normal periodicity gets prohibited
on next samples due to contraction of mov_win from previous sample...need to deal with optimzing window and then keeping
min_window per sensor. Basically, learn the largest window over some time, and then maintain mov_win_min for each sensor

still have despike/interpolation issue on Prod for certain sensor time-series
change mem_window to be 1/4 of primary period instead of primary_period
return only last processed and states data instead of entire state array
rework preiodicity and SNR for case when 95% is too high
also set anomaly points > 50 when initial and after coming back online...
Break out clustering into separate method for async calls in future
Door event excision, correct group performance metric, refine periodicity compute

@description

	Methods for anomaly detection and learning from streaming time-series sensor data:

	1. Data series despiking using Savitsky-Golay filtering, Otsu's method for thresholding, and
		interpolation for spike data replacement. Threshold seeks to maximize inter-class variance
		between "spike" class and "normal" class
	2. Time registration method using interpolation of signals across n-dimensional time-series and optimal sampling rate
	3. ACF, powerspectrum, and DCT filter for time-series periodicity auto-picking and signal feature extraction
	4. dynamic input window size for despike
	5. Periodicity and Power Distance method for sensor metrics
	6. Dynamic Time Warping of time-series for sensor power distance metric
	7. Clustering method of sensor time-series based on Dyanmic Time Warping
	8. Sensor statistical metrics method

@Output : robj = {"data":{},"processed_time":[],"states_time":[],"states":{},"metrics":{}}

	"data" : {<keyname>:[list_of_processed_temp_data_for_keyname],...} for every keyname in set(input sensor_names)
	"processed_time" : [ms_time_value_since_epoch] corresponding to each value returned in time-series under keys "data"
	"states_time" : [ms_time_value_since_epoch] corresponding to each value returned in time-series under keys "states"
	"metrics" : {"cluster_group":{},"group_performance":{},"sensor_performance":{},"group_anomalies":{},"sensor_anomaly":{}}
	"cluster_group" : {<cluster_INT>:[list_of_sensor_names_in_cluster],...} for all cluster_INT in set(clusters)
	"group_performance" : {<cluster_INT>:group_performance_float_value,...} for all cluster_INT in set(clusters)
	"group_anomalies" : {<cluster_INT>:anomaly_true_false_value,...} for all cluster_INT in set(clusters)
	"sensor_performance": {<keyname>:sensor_performance_float_value,...} for every keyname in set(input sensor_names)
	"sensor_anomaly": {<keyname>:anomaly_true_false_value,...} for every keyname in set(input sensor_names)
	"states" : {<various_keys_with_array_states>}

@author michael@glowfish.io
@date 2016-02-10
Copyright (c) 2016__glowfi.sh__. All rights reserved.
"""

#Externals
import numpy as npy
import time
from copy import copy, deepcopy
from scipy.interpolate import griddata, interp1d
from numpy import empty,arange,exp,real,imag,pi
from scipy import signal
from scipy.cluster.hierarchy import fclusterdata, linkage
import pyfftw
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import savgol_filter
import mlpy

pyfftw.interfaces.cache.enable()

#Internals
from pca_anomaly import pcaAnomaly
from utilities import print_timing, heapy

npy.seterr(all='raise')
import logging
glowfish_logger = logging.getLogger('fisherman.lineSinker')

class sensorFlow:

	exceptionClass = None

	def __init__(self, exception):

		self.exceptionClass = exception

		"""
		Time-Series processing, clustering, and anomaly detection class for Glowfish real-time analysis of
		streaming time-series data
		"""

		self.debug = True
		self.small = 1.0e-10
		self.little = 1.0e-10
		self.large = 1.0e+10
		self.larger = 1.0e+32
		self.neg = -1.0e+10


	def sensorProcess(self,sensor_feed,run_cluster,stored_states=None,options=None):
		"""
		:arg sensor_feed: input sensor data as keys with lists of sensor time-series floats
		:arg run_cluster: boolean to determine if reclustering of time-series is done (True)
		:arg stored_states: dictionary containing the previous output ("states" below) of sensorProcess features/decisions
		:arg options: processing flow parameters
		:return: robj :
					"states" : INTERNAL ML STATE KEYS FOR I/O to ML Layer
				 	"data" : processed time-series data for all sensors
				 	"processed_time": array of timestamps associated with processed_data
				 	"states_time" : array of timestamps associated with ML states features
				 	"metrics" : http return from sensorProcess containing current results
		"""

		dbt0 = time.time()

		# Instantiate novelty detection class
		lrner2 = podNovelty(self.exceptionClass)

		# Modifications M. Tompkins 7/4/2016 - added mov_win_init, mov_win_min, and mem_win_min variables below to account
		# for variation in sensor time-series periodicity as it relates to anomaly density (mem_win) and DSP processing
		# window (mov_win)
		# Set some high-level parameters for signal processing and moving windows
		spiker_value = 10.0		# Value to divide despike window by
		run_anomaly = False		# Bool to trigger
		anom_win = 50			# Min states data length to run anomaly detection
		mov_win_init = 3000		# default sample window for sensor metrics
		mov_win_min = 600		# Minimum sample window allowed currently
		mem_win_min = 60		# Minimum sample window for sensor/group anomaly density memory

		# Initialize output object
		robj = {"data":{},"processed_time":[],"states_time":[],"states":{"mavg":{},"primary_period":{},"periods":{},
																		 "periods_indx":{},"snr":{},"powers":{}},
				"metrics":{"cluster_group":{},"group_performance":{},"sensor_performance":{},
						   "group_anomalies":{},"sensor_anomaly":{}}}

		# Define number of current active sensors from input time-series UUID keys
		sensor_list = sensor_feed['data'].keys()

		# get stored time series from states but also initialize for all keys in sensor_list for new sensors that are online
		if stored_states is not None:		# Assign output states from all previous sensors states

			chkpt = False	# Initial call boolean - False here

			states_num = len(stored_states["time_system_performance"])
			raw_num = len(sensor_feed['data'][sensor_list[0]])

			glowfish_logger.debug("Number of processed data input to ML: "+" "+str(states_num))
			glowfish_logger.debug("Number of raw data input to ML: "+" "+str(raw_num))

			robj["states"]["sensor_performance"] = {key : stored_states["sensor_performance"][key] for key in
													stored_states["sensor_performance"].keys()}
			robj["states"]["sensor_deviation"] = {key : stored_states["sensor_deviation"][key] for key in
												  stored_states["sensor_deviation"].keys()}
			robj["states"]["group_anomalies"] = {key : stored_states["group_anomalies"][key] for key in
												  stored_states["group_anomalies"].keys()}
			robj["states"]["powers"] = {key : stored_states["powers"][key] for key in stored_states["powers"].keys()}
			robj["states"]["snr"] = {key : stored_states["snr"][key] for key in stored_states["snr"].keys()}
			robj["states"]["primary_period"] = {key : stored_states["primary_period"][key]
												for key in stored_states["primary_period"].keys()}
			robj["states"]["mavg"] = {key : stored_states["mavg"][key] for key in stored_states["mavg"].keys()}
			robj["states"]["system_performance"] = deepcopy(stored_states["system_performance"])
			robj["states"]["group_performance"] = {key : stored_states["group_performance"][key] for key in
												   stored_states["group_performance"].keys()}
			robj["states"]["sensor_anomaly"] = {key : stored_states["sensor_anomaly"][key] for key in
												stored_states["sensor_anomaly"].keys()}
			robj["states_time"] = deepcopy(stored_states["time_system_performance"])
			robj["states"]["sensor_anomaly_within_groups"] = {key : stored_states["sensor_anomaly_within_groups"][key]
															  for key in stored_states["sensor_anomaly_within_groups"].keys()}

			periods_old = {key : stored_states["periods"][key] for key in stored_states["periods"].keys()}
			periods_indx = {key : stored_states["periods_indx"][key] for key in stored_states["periods_indx"].keys()}

			# Determine if enough points have occured in states time-series to start predicting anomalies
			anom_counter = states_num
			if anom_counter > anom_win:
				run_anomaly = True

				# Modification M. Tompkins 7/4/2016 - added sensor-specific moving_window for metrics computes
				# Handle case where previous primary period was 0 (add 1), and if primary_period << mov_win_init
				# Handle mov_win_init, mov_win_min, and 2.2 x max(previous_period) lower limit to allow for expansion of window
				mov_win = {}
				for key in stored_states["periods_indx"].keys():
					mov_win[key] = mov_win_init
					#try:
					#	mov_win[key] = max([min([int(2.5*(max(stored_states["periods_indx"][key]))),mov_win_init]),mov_win_min])

					#except:
					#	mov_win[key] = mov_win_init

				# Modifications M. Tompkins 7/4/2016 - added sensor-specific anomaly density memory window for alerts
				mem_win = {key : max([stored_states["primary_period"][key][-1]/2,mem_win_min])
							for key in stored_states["primary_period"].keys()}
			else:
				# Modification M. Tompkins 7/4/2016 - added sensor-specific moving_window for metrics computes
				mov_win = {key : mov_win_init for key in stored_states["periods_indx"].keys()}

				# Modifications M. Tompkins 7/4/2016 - added sensor-specific anomaly density memory window for alerts
				mem_win = {key : mem_win_min for key in stored_states["primary_period"].keys()}

			# Determine new and old sensors from states and current sensor_list
			old_keylist = [deviceid for deviceid in stored_states["keylist"] if deviceid in sensor_list]
			old_keylist.extend([deviceid for deviceid in sensor_list if deviceid not in stored_states["keylist"]])
			new_sensors = [deviceid for deviceid in sensor_list if deviceid not in stored_states["keylist"]]
			new_num = len(old_keylist)
			old_num = len(stored_states["keylist"])

			"""
			Initialize new sensor states for all previous timestamps
			This handles case when old sensor goes offline for extended period then back online, since
			stored_states["keylist"] has previous sensor IDs and [new_sensors] comparison above would exclude old sensor
			that was offline. Thus, old sensor coming back online gets initialized states for all of previous time
			(wiped clean) here.
			"""
			for q,deviceid in enumerate(new_sensors):
				robj["states"]["sensor_performance"][deviceid] = [0.0 for p in range(0,states_num)]
				robj["states"]["sensor_deviation"][deviceid] = [0.0 for p in range(0,states_num)]
				robj["states"]["group_anomalies"][str(old_num + q)] = [0.0 for p in range(0,states_num)]
				robj["states"]["powers"][deviceid] = [0.0 for p in range(0,states_num)]
				robj["states"]["snr"][deviceid] = [0.0 for p in range(0,states_num)]
				robj["states"]["mavg"][deviceid] = [0.0 for p in range(0,states_num)]
				robj["states"]["primary_period"][deviceid] = [0.0 for p in range(0,states_num)]
				robj["states"]["group_performance"][str(old_num + q)] = [0.0 for p in range(0,states_num)]
				robj["states"]["sensor_anomaly"][deviceid] = [0.0 for p in range(0,states_num)]
				robj["states"]["sensor_anomaly_within_groups"][deviceid] = [0.0 for p in range(0,states_num)]
				periods_old[deviceid] = []
				periods_indx[deviceid] = []

				# Modification M. Tompkins 7/4/2016 - init moving window for new sensors
				mov_win[deviceid] = mov_win_init

				# Modification M. Tompkins 7/4/2016 - added sensor-specific anomaly density memory window for alerts
				mem_win[deviceid] = mem_win_min

			states_min_old = float(npy.min(robj["states_time"]))
			states_max_old = float(npy.max(robj["states_time"]))

			glowfish_logger.debug("Updated Sensors:"+" "+str(old_keylist))
			glowfish_logger.debug("New Sensors:"+" "+str(new_sensors))
			glowfish_logger.debug("# Total Sensors Now:"+" "+str(new_num))
			glowfish_logger.debug("# Total Groups Now:"+" "+str(robj["states"]["group_anomalies"].keys()))

		else:	# Start fresh and initialize all states time series

			chkpt = True		# Initial call to sensorProcess
			anom_counter = 0
			old_keylist = sensor_list
			robj["states"]["system_performance"] = []
			robj["states"]["group_performance"] = {str(key) : [] for key in range(0,len(sensor_list))}
			robj["states"]["sensor_performance"] = {key : [] for key in sensor_list}
			robj["states"]["sensor_deviation"] = {key : [] for key in sensor_list}
			robj["states"]["powers"] = {key : [] for key in sensor_list}
			robj["states"]["snr"] = {key : [] for key in sensor_list}
			robj["states"]["mavg"] = {key : [] for key in sensor_list}
			robj["states"]["primary_period"] = {key : [] for key in sensor_list}
			robj["states"]["group_anomalies"] = {str(key) : [] for key in range(0,len(sensor_list))}
			robj["states"]["sensor_anomaly_within_groups"] = {}
			robj["states"]["sensor_anomaly"] = {key : [] for key in sensor_list}
			robj["states_time"] = []
			periods_old = {key : [] for key in sensor_list}
			periods_indx = {key : [] for key in sensor_list}
			states_min_old = self.larger
			states_max_old = self.neg
			new_sensors = sensor_list

			# Modification M. Tompkins 7/4/2016 - init moving window for new sensors
			mov_win = {key : mov_win_init for key in sensor_list}

			# Modification M. Tompkins 7/4/2016 - added sensor-specific anomaly density memory window for alerts
			mem_win = {key : mem_win_min for key in sensor_list}

		glowfish_logger.debug("ML - old min and max states times:"+" "+str(states_min_old)+" "+str(states_max_old))

		robj["return"] = True	# Indicate if metrics should be returned via http

		# Time - registration / interpolation of input time-series across all devices for median sample rate of sensors
		data_register, device_min, device_max, delta_t = self.timeRegister(sensor_feed)

		# Depricated code to deal with null sensor data via -999 code replacement
		#data_register, delta_t = self.replaceNullData(sensor_feed)

		# Copy data to DSP window -
		# NOTE: 7/4/2016 - M. Tompkins MAINTAIN mov_win_init for all sensors here due to processed_time equal for all sensors
		data_timed = {"time":npy.asarray(data_register["time"])[-mov_win_init:].tolist(),"data":{}}
		for device_key in data_register["data"]:
			data_timed["data"][device_key] = npy.asarray(data_register["data"][device_key])[-mov_win_init:].tolist()
			glowfish_logger.debug("Moving DSP Window, Anomaly Density Window, Periods: "+str(device_key)+" "+ \
								  str(mov_win[device_key])+" "+str(mem_win[device_key])+" "+str(periods_indx[device_key]))

		robj["processed_time"] = data_register["time"]

		# Assign states_num length of entire processed data stream if starting from scratch
		if chkpt is True:
			states_num = len(robj["processed_time"])

		dbt1 = time.time()
		glowfish_logger.debug("Init Time: "+str(dbt0-dbt1)+" secs")

		# Call periodogram/ACF auto-picking algorithm on signals to produce optimal periods, powerDistance, and SNR metrics
		for device_name in sensor_list:
			fullperiods,fullpower,period,power,snrL = \
				self.getPeriodogram(npy.asarray(data_timed["data"][device_name][-mov_win[device_name]:]),1./delta_t)
			try:
				window_length = int(min(period)/2.)
				if window_length % 2 != 0:
					window_length += 1
			except:
				window_length = 4

			tmp2 = self.getAutocorrelation(npy.asarray(data_timed['data'][device_name][-mov_win[device_name]:]))

			if len(period) == 0:
				primary_period = 0
				if chkpt is True:
					snr_new = 0
					power_dist = 0.0
					tmp0c = []
					periods_new = []
				else:
					snr_new = 0
					if len(periods_old[device_name]) == 0:
						power_dist = 0.0
					else:
						power_dist = npy.linalg.norm(npy.sqrt(npy.sqrt(npy.asarray(periods_old[device_name]))))
					periods_new = deepcopy(periods_old[device_name])
					tmp0c = deepcopy(periods_indx[device_name])
					primary_period = robj["states"]["primary_period"][device_name][-1]
			else:
				options = {"interval":int(min(period)-1)}
				tmp2grad,tmp2gradprime = self.getDerivate(tmp2,window_length)
				if chkpt is True:
					# Check derivatives within window around period candidate and trim based on ACF
					periods_old = None
					tmp0c,index0c = self.trimPeriods(tmp2gradprime,options,period,tmp2)

					try:
						primary_period = tmp0c[npy.argmax(index0c)]
					except:
						primary_period = 0
					power_dist,periods_new,snr_new = self.getPowerDistanceAndSNR(tmp0c,tmp0c,periods_old,
																				   fullperiods,fullpower,snrL)
				else:
					#window_hill = [.5*(2*value+1) - 1,.5*(2*value-1) + 1]
					# Check derivatives within window around period candidate and trim based on ACF
					tmp0c,index0c = self.trimPeriods(tmp2gradprime,options,period,tmp2)

					try:
						primary_period = tmp0c[npy.argmax(index0c)]
						#glowfish_logger.debug("Power and periods "+" "+str(index0c)+" "+str(tmp0c))
					except:
						primary_period = robj["states"]["primary_period"][device_name][-1]
					power_dist,periods_new,snr_new = self.getPowerDistanceAndSNR(tmp0c,periods_indx[device_name],
																				  periods_old[device_name],
																				  fullperiods,fullpower,snrL)

					# Exception when tmp0c is an empty list, re-assign to previous periods_indx
					if len(tmp0c) == 0:
						tmp0c = deepcopy(periods_indx[device_name])

				periods_new = periods_new.tolist()

			# Modification M. Tompkins 6/30/2016 - back-population of current states to all previous states if new sensor
			if device_name in new_sensors:
				robj["states"]["powers"][device_name] = [power_dist for p in range(0,states_num)]
				robj["states"]["snr"][device_name] = [snr_new for p in range(0,states_num)]
				robj["states"]["primary_period"][device_name] = [primary_period for p in range(0,states_num)]

			# Append current state to states
			robj["states"]["primary_period"][device_name].append(primary_period)
			robj["states"]["periods"][device_name] = deepcopy(periods_new)
			robj["states"]["periods_indx"][device_name] = deepcopy(tmp0c)
			robj["states"]["snr"][device_name].append(snr_new)
			robj["states"]["powers"][device_name].append(power_dist)

			# Despike sensor signals
			# Determine despike window from above periods for despike of all time-series
			try:
				sensor_window1 = max( [int(round(min(robj["states"]["periods_indx"][device_name])/spiker_value)),3] )
			except:
				sensor_window1 = 3

			if sensor_window1 % 2 == 0:
				sensor_window1 += 1

			# Adaptive Despike
			# window_val TBD by ACF above
			robj["data"][device_name] = self.seriesDespike(data_register['data'][device_name],sensor_window1)

			# Perform stats metrics calls and pack output
			#robj["states"]["sensor_deviation"][device_name].append(self.getMetrics(npy.asarray(robj["data"][device_name])[-mov_win[device_name]:]))
			stdev_tmp,mavg_tmp = self.getMetrics(npy.asarray(robj["data"][device_name])[-mov_win[device_name]:])

			# Modification M. Tompkins 6/30/2016 - back propagation of current states to all previous states if new sensor
			if device_name in new_sensors:
				robj["states"]["mavg"][device_name] = [mavg_tmp for p in range(0,states_num)]
				robj["states"]["sensor_deviation"][device_name] = [stdev_tmp for p in range(0,states_num)]

			robj["states"]["sensor_deviation"][device_name].append(stdev_tmp)
			robj["states"]["mavg"][device_name].append(mavg_tmp)

			glowfish_logger.debug("DeviceID and Length of States: "+device_name+" "+str(len(robj["states"]["mavg"][device_name])))

		dbt2 = time.time()
		glowfish_logger.debug("Sensor Health and Despike Time:"+str(dbt2-dbt1)+" secs")

		# The data window below is all input data and is different than period window above mov_win[device_name]

		# If first call to sensorProcess, then run DTW and clustering, else use previous until run_cluster == True
		if chkpt is True:

			glowfish_logger.debug("Initial Cluster Run:"+str(run_cluster))

			# Dynamic Time Warping and Clustering and group metrics
			distmat,keylist = self.getDTWDistance(robj["data"])
			robj["states"]["keylist"] = keylist		# Hold well-ordered list of UUIDs for subsequent coherence in cluster groups

			# Edge case handling for single sensor feed
			if len(sensor_list) == 1:
				groups = {"0":sensor_list}
			else:
				groups = self.getSensorClusters(distmat,keylist)

			# Initialize output groups and performance arrays for all sensors UUIDs (#clusters are ALWAYS <= #sensors)
			for groupname in groups:
				robj["states"]["group_performance"][groupname] = []
				robj["metrics"]["group_performance"][groupname] = []
				robj["states"]["group_anomalies"][groupname] = []
				robj["metrics"]["group_anomalies"][groupname] = []
				for device3 in groups[groupname]:
					robj["states"]["sensor_anomaly_within_groups"][device3] = []

		elif run_cluster is True:				# Run clustering if enough time has passed and triggered by API

			glowfish_logger.debug("Cluster Run:"+str(run_cluster))

			# Dynamic Time Warping and Clustering and group metrics
			distmat,keylist = self.getDTWDistance(robj["data"],old_keylist)
			robj["states"]["keylist"] = keylist

			# Edge case handling for single sensor feed
			if len(sensor_list) == 1:
				groups = {"0":sensor_list}
			else:
				groups = self.getSensorClusters(distmat,keylist)

		else:						# Dont re-cluster just use previous cluster_group state but pop any offline sensors
			groups = {}
			for group_name in stored_states["cluster_group"]:
				groups[group_name] = [i for i in stored_states["cluster_group"][group_name]
									  if i in old_keylist]
			robj["states"]["keylist"] = old_keylist

		dbt3 = time.time()

		robj["metrics"]["cluster_group"] = deepcopy(groups)
		robj["states"]["cluster_group"] = deepcopy(groups)

		glowfish_logger.debug("Clustering Time: "+str(dbt3-dbt2)+" secs")
		glowfish_logger.debug("Current Groups: "+str(robj["states"]["cluster_group"].keys()))

		# Run subspace projection and anomaly detection if run_anomaly == True
		if run_anomaly is True:

			tmp_system = []

			# Run P.O.D. and anomaly detection on each sensor feature space individually
			for device_name in sensor_list:

				# First determine if sensor time-series has defined periodicity, else return 0.0 result for anomalies
				if npy.max(robj["states"]["snr"][device_name]) > self.small and npy.max(robj["states"]["powers"][device_name]) > self.small:
					anom = lrner2.podDetect({"data":{"snr":robj["states"]["snr"][device_name],
							"deviation":robj["states"]["sensor_deviation"][device_name],
							"power":robj["states"]["powers"][device_name]},"mavg":robj["states"]["mavg"][device_name]})
				# removed primary period from anomly detect - 6/23/2106
				# "primary_period":robj["states"]["primary_period"][device_name]})
				else:
					anom = {"anomaly_probability" : [0.0]}


				# Assign actual anomaly predictions based on probabilities returned from detection class and previous states
				tmp_sensor2 = 0.0

				if anom["anomaly_probability"][-1] > 0.67 and anom["anomaly_probability"][-1] < 0.97:

					tmp_sensor2 = 1.0
				elif anom["anomaly_probability"][-1] >= 0.97:

					tmp_sensor2 = 2.0

				# Set current window for anomaly density (i.e., performance) to be computed over mem_win parameter
				current_win = min([len(robj["states"]["sensor_performance"][device_name]),mem_win[device_name]])

				if anom_counter == anom_win + 1:	# First call to anomaly detect
					mean_sense = float(npy.mean(npy.asarray(robj["states"]["sensor_anomaly"][device_name])[-current_win:]))
					max_val_sense =	mean_sense + (tmp_sensor2-mean_sense)/(current_win + 1)
				else:
					mean_sense = float(npy.mean(npy.asarray(robj["states"]["sensor_anomaly"][device_name])[-current_win:]))
					max_val_sense =	mean_sense + (tmp_sensor2-mean_sense)/(current_win + 1)

				glowfish_logger.debug("sensor_performance"+" "+device_name+" "+str(max_val_sense)+" "+str(current_win)+" "+str(mean_sense))

				# Assign output values for anomalies in metrics and states return
				robj["metrics"]["sensor_anomaly"][device_name] = tmp_sensor2
				robj["states"]["sensor_anomaly"][device_name].append(tmp_sensor2)
				robj["metrics"]["sensor_performance"][device_name] = max_val_sense
				robj["states"]["sensor_performance"][device_name].append(max_val_sense)
				tmp_system.append(max_val_sense)

				glowfish_logger.debug("sensor memory window"+" "+device_name+" "+str(current_win))

			# Now run subspace projection and anomaly detection on each group of sensors
			groupnew = {}
			for groupname2 in robj["metrics"]["cluster_group"]:
				groupnew[groupname2] = {"data":{}}
				for device1 in robj["metrics"]["cluster_group"][groupname2]:
					groupnew[groupname2]["data"][device1] = robj["data"][device1]

				groupnew[groupname2].update(lrner2.podDetect(groupnew[groupname2]))

				groupnew[groupname2].pop("anomaly_predictions")

				tmp_anomaly2 = 0.0
				if groupnew[groupname2]["anomaly_probability"][-1] > 0.67 and groupnew[groupname2]["anomaly_probability"][-1] < 0.97:

					tmp_anomaly2 = 1.0

				elif groupnew[groupname2]["anomaly_probability"][-1] >= 0.97:

					tmp_anomaly2 = 2.0

				if anom_counter == anom_win + 1:	# First pass through anomaly detect for cluster groups

					#max_val_anom = tmp_anomaly2
					#tmp_system.append(max_val_anom)
					group_sense = float(npy.mean(npy.asarray(robj["states"]["group_anomalies"][groupname2])[-current_win:]))
					max_group_sense = group_sense + (tmp_anomaly2 - group_sense)/(current_win + 1)

				elif len(robj["states"]["group_performance"][groupname2]) == 0:	# No states anomaly data to process

					max_group_sense = tmp_anomaly2

				else:

					#max_val_anom = int(max([tmp_anomaly2,npy.max(npy.asarray(robj["states"]["group_anomalies"][groupname2])[-current_win:])]))
					group_sense = float(npy.mean(npy.asarray(robj["states"]["group_anomalies"][groupname2])[-current_win:]))
					max_group_sense = group_sense + (tmp_anomaly2 - group_sense)/(current_win + 1)

				tmp_system.append(max_group_sense)

				# Assign output values for cluster group anomalies in metrics and states return
				robj["metrics"]["group_anomalies"][groupname2] = tmp_anomaly2
				robj["states"]["group_anomalies"][groupname2].append(tmp_anomaly2)
				robj["states"]["group_performance"][groupname2].append(float(max_group_sense))
				robj["metrics"]["group_performance"][groupname2] = float(max_group_sense)

				glowfish_logger.debug("group performance"+" "+str(groupname2)+" "+str(robj["metrics"]["group_performance"][groupname2]))

				# Propagate group anomalies down to each sensor individually for UI and QA plotting purposes
				for device2 in robj["metrics"]["cluster_group"][groupname2]:
					robj["states"]["sensor_anomaly_within_groups"][device2].append(tmp_anomaly2)

			# Assign entire system performance based on group and sensor statuses (anomalies)
			robj["metrics"]["system_performance"] = max(tmp_system)
			robj["states"]["system_performance"].append(robj["metrics"]["system_performance"])

			# Append a 0 placeholder in states arrays for any group ID that doesn't currently have a sensor assigned to it
			for groupname3 in robj["states"]["group_performance"]:
				if groupname3 not in robj["metrics"]["cluster_group"]:
					robj["states"]["group_performance"][groupname3].append(0.0)

			dbt4 = time.time()
			glowfish_logger.debug("Sensor and Group Anomaly Detect Time:"+str(dbt4-dbt3)+" secs")

		elif chkpt is True:			# Back-populate performance for all processed times
			robj["states"]["system_performance"] = [0.0 for p in range(states_num)]
			robj["metrics"]["system_performance"] = [0.0 for p in range(states_num)]
			for device_name in sensor_list:
				robj["states"]["sensor_anomaly"][device_name] = [0.0 for p in range(states_num)]
				robj["metrics"]["sensor_performance"][device_name] = 0.
				robj["states"]["sensor_performance"][device_name] = [0.0 for p in range(states_num)]
			for groupname2 in robj["metrics"]["cluster_group"]:
				robj["states"]["group_anomalies"][groupname2] = [0.0 for p in range(states_num)]
				robj["states"]["group_performance"][groupname2] = [0.0 for p in range(states_num)]
				robj["metrics"]["group_performance"][groupname2] = 0.
				for device2 in robj["metrics"]["cluster_group"][groupname2]:
					robj["states"]["sensor_anomaly_within_groups"][device2] = [0.0 for p in range(states_num)]
			for groupname3 in robj["states"]["group_performance"]:
				if groupname3 not in robj["metrics"]["cluster_group"]:
					robj["states"]["group_performance"][groupname3] = [0.0 for p in range(states_num)]
					robj["states"]["group_anomalies"][groupname3] = [0.0 for p in range(states_num)]

		else:		# Append a 0 placeholder anomaly in system, group, and sensor performances until anomaly detect is run
			robj["states"]["system_performance"].append(0.0)
			robj["metrics"]["system_performance"] = 0.0
			for device_name in sensor_list:
				robj["states"]["sensor_anomaly"][device_name].append(0.0)
				robj["metrics"]["sensor_performance"][device_name] = 0.
				robj["states"]["sensor_performance"][device_name].append(0.)
			for groupname2 in robj["metrics"]["cluster_group"]:
				robj["states"]["group_anomalies"][groupname2].append(0.0)
				robj["states"]["group_performance"][groupname2].append(0.)
				robj["metrics"]["group_performance"][groupname2] = 0.
				for device2 in robj["metrics"]["cluster_group"][groupname2]:
					robj["states"]["sensor_anomaly_within_groups"][device2].append(0.0)
			for groupname3 in robj["states"]["group_performance"]:
				if groupname3 not in robj["metrics"]["cluster_group"]:
					robj["states"]["group_performance"][groupname3].append(0.)
					robj["states"]["group_anomalies"][groupname3].append(0.0)

		# Append states time array separately from processed data time array due to offset between states and processed
		# data at beginning of stream
		if chkpt is True:				# Start from scratch, so create entire processed data time series from all data
			robj["states_time"] = robj["processed_time"]
		else:							# Append last time stamp for states
			robj["states_time"].append(robj["processed_time"][-1])

		# Now determine largest of states timerange from input and new interpolate states output for DB deletion
		robj["min_states_timestamp"] = min([states_min_old,float(npy.min(robj["states_time"]))])
		robj["max_states_timestamp"] = max([states_max_old,float(npy.max(robj["states_time"]))])

		glowfish_logger.debug("ML - new min and max states times:"+" "+str(robj["min_states_timestamp"])+" "+str(robj["max_states_timestamp"]))
		glowfish_logger.debug("sensor performance length"+" "+str(len(robj["states"]["system_performance"])))
		glowfish_logger.debug("group performance length"+" "+str(len(robj["states"]["group_performance"][str(0)])))

		return robj


	def signalExtract(self,datahash,options=None):

		"""
		Interface for seriesProcess seriesDespike method
		If options=None, only despike is performed with default window
		else, options is a dictionary containing glowfish flow options:
		window = [odd integer required]

		:arg datahash input sensor time-series raw data feed dictionary
		:return depiked sensor time-series data feed dictionary
		"""

		try:
			# Currently despike is implemented
			window_val = None
			if options is not None:
				if 'window'in options:
					window_val = options['window']

			for key3 in datahash['data']:
				datahash['data'][key3] = self.seriesDespike(datahash['data'][key3],window_val)

		except Exception as e:
			glowfish_logger.exception(e)
			raise self.exceptionClass('460.2')

		return datahash


	def replaceNullData(self,data_in):

		"""
		N-dim data series replacement of -999 null data from API by linear interpolation.

		:arg sensor time-series raw data
		:return sensor time-series data with -999 replaced by interpolated values
		"""

		try:

			dataN = deepcopy(data_in)
			datanew = {'data':{},'time':dataN["time"]}

			# Index any samples == -999 for each sensor
			for m,keys in enumerate(dataN['data']):
				samples = range(0,len(dataN['data'][keys]))
				times = npy.asfarray(dataN['time'])
				tmp_array = npy.zeros(shape=(len(samples)),dtype=float)
				inx_null = [g for g in samples if dataN['data'][keys][g] - (-999.) <= self.small]
				inx_ok = [g for g in samples if dataN['data'][keys][g] - (-999.) > 1]

				if len(inx_null) == 0:	# No NULL values in array
					tmp_array[:] = dataN['data'][keys]

				elif len(inx_ok) == 1:	# Only 1 good value in array
					tmp_array[:] = dataN['data'][keys][inx_ok[0]]

				elif len(inx_ok) == 0:	# No GOOD values in array so maintain nulls
					tmp_array[:] = -999.

				else:					# At least 2 good values in array to interpolate

					# Handle case of first point to set lower bound for interpolation
					if inx_null[0] == 0:
						dataN['data'][keys][0] = dataN['data'][keys][inx_ok[0]]

					# Handle case of last point to set upper bound for interpolation
					if inx_null[-1] == len(dataN['data'][keys])-1:
						dataN['data'][keys][-1] = dataN['data'][keys][inx_ok[-1]]

					# Re-index Null values with lower/upper bounds set from above
					inx_mod = [g for g in samples if dataN['data'][keys][g] - (-999.) > 1]

					# Handle interior points with cubic spline interpolant
					signal_intrp = interp1d(times[inx_mod],npy.asfarray(dataN['data'][keys])[inx_mod]) #,kind='cubic')
					tmp_array = signal_intrp(times)

				datanew['data'][keys] = tmp_array.tolist()

			# Compute average sampling rate for all times
			device_delta_t = float(npy.mean(npy.asfarray(datanew['time'][1:]) - npy.asfarray(datanew['time'][0:-1])))

			return datanew, device_delta_t

		except Exception as e:
				glowfish_logger.exception(e)
				raise self.exceptionClass('461.2')


	def seriesDespike(self,dataN,window_opt=None):

		"""
		Data series despiking using Savitsky-Golay filtering, Otsu's method for thresholding, and
		interpolation for spike data replacement

		:arg list : sensor time-series
		:return list : de-spiked data of same length as input time-series array

		IMPORTANT : GlowfishException disabled here to avoid catastrophic failure of ML if signal has zero variance
				If data has zero variance, then input time-series list is returned
		"""

		try:

			# Define spike window and interpolate signal with 2nd order polynomial
			if window_opt is not None:
				winlen = window_opt
			else:
				winlen = 5	# must be an odd integer of order of expected spike sample width x (1.5 - 3)

			padlen = 1

			# Copy of input signal for spike replacement
			sigint2 = npy.asfarray(dataN)

			# Pad the signal by mean of previous 20-sample signal above global time-series mean value to
			# avoid poor interpolation at the endpoint of array
			sigint = npy.ndarray(shape=(len(sigint2)+padlen),dtype=float)
			sigint[0:len(sigint2)] = sigint2

			#Add Datuming to check for both positive spikes AND negative spikes
			datum = npy.abs(sigint2[-20:-1]) - npy.max(npy.abs(sigint2[-20:-1]))
			tq = npy.mean(datum)
			datavg = npy.mean(sigint2[-20:-1])

			paddarr = sigint2[-20:-1]

			padmask = (datum <= tq)
			if len(datum[padmask]) == 0:
				padval = datavg
			else:
				padval = npy.mean(paddarr[padmask])

			sigint[len(sigint2):] = padval	# Ignore last point for padding mean computation

			# Interpolate the time-series with degree-2 polynomial
			signalnew = savgol_filter(sigint,winlen,2)

			# Compute (raw signal - interpolated signal) residuals for statistical Otsu's test
			diffsig = npy.abs(sigint-signalnew)
			diffperc = diffsig*(1/(npy.abs(npy.mean(sigint))))

			# Determination of the most appropriate threshold for the spike detection using Otsuʼs method
			# Histogram binning of the signal residual and initialization of the threshold
			[counts1,edges]=npy.histogram(diffsig,100)
			amp_range = [edges[m] for m in range(1,len(edges))]
			counts1 = counts1 * (1/float(len(diffsig)))

			# Calculation of the initial class probabilities and means
			proba_C1 = npy.ndarray(shape=len(amp_range),dtype=float)
			proba_C2 = npy.ndarray(shape=len(amp_range),dtype=float)
			mean_C1 = npy.ndarray(shape=len(amp_range),dtype=float)
			mean_C2 = npy.ndarray(shape=len(amp_range),dtype=float)
			C1C2_var = npy.ndarray(shape=len(amp_range)-1,dtype=float)
			threshold = npy.ndarray(shape=len(amp_range)-1,dtype=float)

			proba_C1[0] = counts1[0]

			if proba_C1[0] < self.small:
				proba_C2[0] = 0.0
				mean_C1[0] = 0.0
				mean_C2[0] = 0.0
			else:
				proba_C2[0] = 1 - proba_C1[0]
				mean_C1[0] = (amp_range[0]*counts1[0])/proba_C1[0]
				mean_C2[0] = npy.sum(amp_range[1:len(amp_range)]*counts1[1:len(amp_range)])/proba_C2[0]

			# Test all possible threhold values and their effect on the between-class variance
			threshold[0] = amp_range[1]
			C1C2_var[0] = 0.0
			for i in range(1,len(amp_range)-1):
					threshold[i] = amp_range[i+1]
					proba_C1[i] = npy.sum(counts1[0:i])
					proba_C2[i] = 1-proba_C1[i]
					mean_C1[i] = npy.sum(amp_range[0:i]*counts1[0:i])/proba_C1[i]
					mean_C2[i] = npy.sum(amp_range[i+1:len(amp_range)]*counts1[i+1:len(amp_range)])/proba_C2[i]
					C1C2_var[i] = proba_C1[i]*proba_C2[i]*(mean_C1[i]-mean_C2[i])**2

			# The optimal threshold maximizes the between-class variance
			#indx = len(C1C2_var) + 1 - npy.argmax(C1C2_var[:0:-1])
			indx = npy.argmax(C1C2_var)

			#Handle case where NO spike is not greater than 5% of interpolated signal and ignore threshold
			if npy.max(diffperc) < .05:

				signalout = sigint			# Output raw data

			else:		# Output despiked signal

				finalthresh = threshold[indx]

				# Mask and index array elements that are above threshold
				mask = diffsig > finalthresh
				x = [m for m in range(0,len(mask)) if mask[m] == True]

				# Set up indexing arrays for defining spikes in time-series
				startspike = x[0]
				spikes = npy.ndarray(shape=len(x),dtype=int)
				jmin = npy.ndarray(shape=len(x),dtype=int)
				jmax = npy.ndarray(shape=len(x),dtype=int)

				# Loop over spikes for first N-1 points and replace with interpolated values
				i = 0
				for k in range(0,len(x)-1):

					#case of consecutive points belonging to the same spike
					if x[k+1]-x[k] <= 1:
						k += 1

					# case of a separate spike
					else:
						stopspike = x[k]
						spikes[i] = npy.floor((stopspike+startspike)/2.)
						startspike = x[k+1]

						#Step 2: excision of the spike region, s, of width Wspike (centered on
						# the spike index and comprised between jmin and jmax) from the original signal
						# by linear interpolation.
						jmin[i] = max(1,spikes[i] - (winlen - 1)/2)

						jmax[i] = min(spikes[i] + (winlen - 1)/2,len(sigint)-1)
						s = range(jmin[i],jmax[i] + 1)
						sigint[s]=sigint[jmin[i]] + (sigint[jmax[i]] - sigint[jmin[i]])/(jmax[i] - jmin[i])*(s - jmin[i] + 1)
						i += 1

				samples = i

				# Deal with last spike in time-series
				stopspike = x[-1]
				spikes[i] = npy.floor((stopspike+startspike)/2)
				jmin[i] = min(spikes[i] - (winlen - 1)/2,len(sigint)-1)
				jmax[i] = min(spikes[i] + (winlen - 1)/2,len(sigint)-1)

				s = range(jmin[i],jmax[i] + 1)
				sigint[s]=sigint[jmin[i]]+(sigint[jmax[i]]-sigint[jmin[i]])/(jmax[i] - jmin[i])*(s - jmin[i] + 1)

				# Smooth the interpolated spike regions and replace spikes in the original series
				signalout = copy(sigint)

				tmpsig = savgol_filter(sigint,winlen,2)

				# Final replacement of spike regions for output
				for i in range(0,samples):
					s = range(jmin[i],jmax[i])
					signalout[s] = tmpsig[s]

		except Exception as e:
				glowfish_logger.exception(e)
				return dataN
				#raise self.exceptionClass('461.2')

		return signalout[0:-padlen].tolist()


	def timeRegister(self,dataN,update_dict=None):

		"""
		N-dim data series time registration using interpolation. Interpolation made to mean
		sampling rate of n-dim sensor time-series with outliers removed before mean sample rate is computed.

		:arg dataN : input hash with all current sensor time-series to be interpolated
		:return datanew : interpolated time-series hash with additional key "time" representing the new timestamp array for all sensors
				device_min : minimum of device sample rates
				device_max : maximum of device sample rates
				device_delta_t : the resulting mean time delta (sampling rate) for all sensor time-series returned
		"""

		try:

			datanew = {'data':{},'time':{}}
			dims2 = len(dataN['data'].keys())

			key_order = []
			min_sample = 0.0
			min_start = self.larger
			max_start = 0.0

			# Determine high/low time range and lowest sampling rate to perform registration over all dims
			# Throw away extreme sample rates 95th percentile for sample_rate computation only
			# Throw away interpolated data outside range of all feeds individually
			# TODO throw away interpolated data when samples are too far apart based on outlier sample rate above
			mask = {}
			device_min = {}
			device_max = {}

			for m,keys in enumerate(dataN['time']):

				key_order.append(keys)
				timeval = npy.sort(npy.asarray(dataN['time'][keys]))
				device_min[keys] = npy.min(timeval)
				device_max[keys] = npy.max(timeval)
				min_start = npy.min([npy.min(timeval),min_start])
				max_start = npy.max([npy.max(timeval),max_start])
				all_samples = timeval[1:] - timeval[0:-1]
				max_sample_rate = npy.percentile(all_samples,95)
				mask[keys] = all_samples <= max_sample_rate
				#[vals,bins] = npy.histogram(all_samples[mask],100)
				#min_sample = npy.max([min_sample,round(bins[npy.argmax(vals)],2)])
				min_sample = npy.max([min_sample,npy.mean(all_samples[mask[keys]])])
				#median_sampling = npy.max([median_sampling,round(bins[npy.median(vals)],2)])

			# TODO revisit down sampling below
			#min_sample *= 2.
			dims1a = len(npy.arange(min_start,max_start,min_sample))
			signew = npy.ndarray(shape=(dims1a,dims2),dtype=float)
			samples = npy.arange(min_start,max_start,min_sample)
			device_delta_t = min_sample

			# Interpolate all samples to mean sampling rate
			# Excise any new time-series samples outside sensor max, min, max_sample_rate
			for m,keys in enumerate(dataN['data']):
				sigint = npy.asarray(dataN['data'][keys])
				timeval = npy.asarray(dataN['time'][keys])
				keynew = interp1d(timeval,sigint)

				# Assign mean to output array, then fill in interpolated values between min,max time support indexes
				signew[:,m] = npy.mean(sigint)
				mask_tmp1 = samples < device_min[keys]
				mask_tmp2 = samples > device_max[keys]
				try:
					sample_indx1 = samples.tolist().index(samples[mask_tmp1][-1]) + 1	#Added 1 here for edge case 6/23/2016

				except:
					sample_indx1 = 0
				try:
					sample_indx2 = samples.tolist().index(samples[mask_tmp2][0]) - 1	#Subtractred 1 here for edge case 6/23/2106
				except:
					sample_indx2 = len(samples)-1

				# Replace trailing and leading edges with appropriate original trailing and leading data - NEW 6/23/2016
				signew[sample_indx2:,m] = sigint[-1]
				signew[0:sample_indx1] = sigint[0]

				# Interpolate remaining time support - changed indexing by -1 on sample_indx1 here for edge case - 6/23/2016
				signew[sample_indx1:sample_indx2+1,m] = keynew(samples[sample_indx1:sample_indx2+1])

				# Listify dictionary arrays for return
				datanew['data'][keys] = signew[:,m].tolist()

			# Move time samples to single key since they are all the same now
			datanew['time'] = samples.tolist()

			return datanew, device_min, device_max, device_delta_t

		except Exception as e:
				glowfish_logger.exception(e)
				raise self.exceptionClass(e) #'462.2')


	def dctCompute(self,data_in,options="forward"):

		"""
		1-D data series DCT/IDCT utility method
		:arg input data time-series
		:return dct or idct coefficients of same length
		"""

		try:
			datadct = data_in.copy()
			N = len(datadct)

			if options == "forward":
				# Perform Forwrard DCT
				dctout = self.dct(datadct)
				dctout[0] = dctout[0]*npy.sqrt(1/(4.*N))
				dctout[1:] = dctout[1:]*npy.sqrt(1/(2.*N))

				return dctout

			elif options == "inverse":
				# Perform Inverse DCT
				idcttmp = datadct
				idcttmp[0] = idcttmp[0]/npy.sqrt(1/(4.*N))
				idcttmp[1:] = idcttmp[1:]/npy.sqrt(1/(2.*N))
				idctout = self.idct(idcttmp)

				return idctout

		except Exception as e:
				glowfish_logger.exception(e)
				raise self.exceptionClass('461.2')


	def dct(self,y):

		"""
		Forward DCT utility method implemented with FFTW
		"""

		N = len(y)
		y2 = empty(2*N,float)
		y2[:N] = y[:]
		y2[N:] = y[::-1]
		t = pyfftw.builders.rfft(y2)
		c = t()
		phi = exp(-1j*pi*arange(N)/(2*N))

		return real(phi*c[:N])


	def idct(self,a):

		"""
		Inverse DCT untility method implemented with FFTW
		"""

		N = len(a)
		c = empty(N+1,complex)

		phi = exp(1j*pi*arange(N)/(2*N))
		c[:N] = phi*a
		c[N] = 0.0
		t = pyfftw.builders.irfft(c)
		d = npy.real(t()[:N])

		return d


	def getAutocorrelation(self,data_in,options=None):

		"""
		Compute auto-correlation function for a 1-D array used in conjunction with power spectrum to solve for optimal
		periodicity in time-series window

		:arg data_in : input time-series array
		:return acfout : autocorrelation coefficients of length of input array
		"""

		try:
			n = len(data_in)
			#nlags = n/10
			#variance = data_in.var()
			x = data_in-data_in.mean()
			r = npy.correlate(x, x, mode = 'full')[-n:]
			acfout = r/npy.max(r)

			return acfout

		except Exception as e:
				glowfish_logger.exception(e)
				raise self.exceptionClass('463.2')


	def getDerivate(self,data_in,options=None):

		"""
		Utility method for 1st and 2nd derivatives of discrete 1D function by central differences and then interpolate
		to original dimensions to solve for optimal periodicity from ACF and power spectrum methods

		:arg data_in : input 1D function series
		:return grad_out : array of first central differences of input function
				prime_out : array of second central differences of input function
		"""

		try:

			if options is not None:
				win2 = options
				intvl = 1
			else:
				win2 = 4
				intvl = 1

			# Compute gradient by central differences over intvl line segments
			samples1 = range(0,(len(data_in)-intvl))
			samples2 = range(0,len(data_in),intvl)

			grad = npy.ndarray(shape=len(samples2),dtype=float)
			grad_prime = npy.ndarray(shape=len(samples2),dtype=float)
			grad[0] = 0.0			#Start point
			grad[-1] = 0.0			#End point
			grad_prime[0] = 0.0		#Start point
			grad_prime[-1] = 0.0	#End point

			data_inF = npy.zeros(shape=len(samples2),dtype=float)

			data_inF[0:win2/2] = data_in[0:win2/2]

			data_inF[-win2/2:-1] = data_in[-win2/2:-1]

			data_inF[win2/2:-win2/2] = [npy.mean(data_in[i-win2:i]) for i in range(win2,len(samples2))]

			# Compute first and second central differences of series
			grad[1:-2] = 1/2. * (data_in[0:-3:intvl]-data_in[2:-1:intvl])
			grad_prime[1:-2] = 1/1. * (data_inF[0:-3:intvl]+data_inF[2:-1:intvl]-2.*data_inF[1:-2:intvl])

			# Interpolate to original sampling
			prime_intr = interp1d(samples2,grad_prime)
			grad_intr = interp1d(samples2,grad)
			grad_out = grad_intr(samples1)
			prime_out = prime_intr(samples1)

			return grad_out,prime_out

		except Exception as e:
			glowfish_logger.exception(e)
			raise self.exceptionClass('464.2')


	def getPeriodogram(self,series_in,delta_t,options=None):
		"""
		Compute periodogram of discrete function and return signal to noise, powers, and dominant periods. Used in
		conjunction with ACF and gradient methods to optimize signal periodicity.

		:arg series_in : input time-series 1D array to compute periodogram
		:return periods : array of all spatial periods derived from input
				Pxx_den : array of all powers (amplitudes) for periods in input
				periods[Pmask][Pmaskf] : array of only the dominant periods with signal above 95% noise percentile
				Pxx_den[Pmask][Pmaskf] : array of only the dominant powers (amplitudes) with signal above 95% noise percentile
				s_noise_l : estimated SNR for input time-series dominant periods
		"""

		try:

			# Run permuted (randomized sampling) periodogram for noise level estimation of time-seriesand set 95%
			# threshold for signal series_norm = (series_in-npy.mean(series_in))*(1/npy.std(series_in))
			max_power = []
			for i in range(0,100):
				Qp = npy.random.permutation(series_in)
				ftmp, Ptmp = signal.periodogram(Qp)
				max_power.append(npy.percentile(Ptmp,95))
			thresh = npy.percentile(max_power,99)

			# Compute actual periodogram from well-ordered time-series
			f, Pxx_den = signal.periodogram(series_in)

			# Mask powers above noise theshold
			Pxx_den[0] = 0.0001
			Pmask = Pxx_den > thresh

			# Compute periods from frequencies
			periods = [len(series_in)+1]
			periods.extend([1./fp for fp in f[1:]])

			# Remove large periods
			Pmaskf = npy.asarray(periods)[Pmask] < len(series_in)/2.
			noise = (1/npy.percentile(max_power,50))
			s_noise_l = Pxx_den*noise

			return npy.asarray(periods),Pxx_den,npy.asarray(periods)[Pmask][Pmaskf],Pxx_den[Pmask][Pmaskf],s_noise_l

		except Exception as e:
			glowfish_logger.exception(e)
			raise self.exceptionClass('465.2')


	def getPowerDistanceAndSNR(self,tmp0a,periods_indx,periods_old,fullperiods,fullpower,snrL):
		"""
		Compute power distance between previous and current periodicity of discrete function by interpolation of current
		periodogram series to ACF time periods and computing L2 norm of previous state and current state.
		Also convert input SNR to units: dB.

		:arg tmp0a (list of dominant periods), periods_indx (ist of indexes of dominant periods within entire power series)
				periods_old (previous periods state computed),fullperiods (full array of periods),fullpower (full array of powers)
				snrL (input SNR ratio computed from getPeriodogram())
		:return power_new (computed power distance from previous to current state)
				periods_new (current dominant periodocity list)
				snr_new : current SNR in dB
		"""

		try:

			#Exceptions if no peak period found or begining from first call in time-series
			if periods_old is None:

				if len(tmp0a) == 0:

					snr_new = 0.0
					period_intr = interp1d(fullperiods,fullpower)
					periods_new = period_intr(periods_indx)
					power_dist = 0.0

				else:

					# Interpolate periodicity and SNR series to current ACF time lags for comparison with previous state
					period_intr = interp1d(fullperiods,fullpower)
					periods_new = period_intr(tmp0a)
					snr_intr = interp1d(fullperiods,snrL)

					#dB conversion of SNR
					snr_new = npy.log10(npy.mean(snr_intr(tmp0a)))*20.0
					power_dist = 0.0

			else:

				if len(tmp0a) == 0:

					snr_new = 0.0
					period_intr = interp1d(fullperiods,fullpower)
					periods_new = period_intr(periods_indx)
					power_dist = npy.linalg.norm(npy.sqrt(npy.sqrt(npy.asarray(periods_old))))

				else:

					# Interpolate periodogram to currrent ACF time lags for comparison with previous state
					period_intr = interp1d(fullperiods,fullpower)
					periods_new = period_intr(tmp0a)
					snr_intr = interp1d(fullperiods,snrL)

					#dB conversion of SNR
					snr_new = npy.log10(npy.mean(snr_intr(tmp0a)))*20.0
					#print npy.linalg.norm(npy.sqrt(npy.asarray(periods_new))-npy.sqrt(npy.asarray(periods_old)))
					power_dist = npy.linalg.norm(npy.sqrt(npy.asarray(period_intr(periods_indx))) \
													 -npy.sqrt(npy.asarray(periods_old)))

			power_new = power_dist

			return power_new,periods_new,snr_new

		except Exception as e:
			glowfish_logger.exception(e)
			raise self.exceptionClass(e) #'466.2')

	@print_timing
	def trimPeriods(self,tmp2gradprime,options,period,tmp2):

		"""
		Map candidate dominant time-series periods returned by getPeriodogram() to ACF time lags, then determine by first
		and second derivatives of ACF whether these candidates map to maxima within a local window of ACF series, if so,
		segment window between +- 1/10 * period to estimate nearest maximum and assign new periods to these ACF maxima.

		Method : "On Periodicity Detection and Structural Periodic Similarity", Vlachos, Yu, & Castelli, 2005

		:arg tmp2gradprime : first and second derivatives of ACF returned from getDerivative()
				options : base interval of periodicity of time-series to be used as local ACF window here
				period : candidate periods list from getPeriodogram()
				tmp2 : list of indexes of dominant periods series
		:return tmp0c : list of "trimmed" periods translated to represent ACF maxima not periodogram maxima
				indx_trim2 : list of "trimmed" powers for "trimmed" periods in entire periodicity array
		"""

		try:

			tmp1 = []
			tmp1a = []
			tmp0 = []
			tmp0a = []
			Np = len(tmp2)

			# Iterate over all input dominant periods returned by getPeriodogram()
			tmp8 = []
			tmp8a = []
			for value in period:

				# Set local ACF windows for determining if derivatives are maximal locally
				wind1 = range(max([int(value) - int(round(value/8.,0)),int(round(min(period),0))]),
				  int(value) + int(round(value/2.,0)))

				wind2 = range(max([int(value) - int(round(min(period)/2.,0)),int(round(min(period)/2.,0))]),
				  max([int(value) + int(round(value/20.,0)),int(value) + int(round(min(period)/2.,0))]))

				err_chk = 1.0e+10
				valid = False
				loc = None
				for k in range(min([18,len(wind2)-2])):	# Perform 10-bisection regression to get best split of window
					a = wind2[0]
					b = wind2[-1]
					split = max([len(wind2)/20,1])
					c = wind2[0] + (split*(k+1))
					#print int(value),a,b,c,split,k
					obsT = npy.vstack([npy.asarray(tmp2[a:c]),npy.ones(len(tmp2[a:c]))]).T
					#print obsT.shape, npy.asfarray(range(a,c))
					[slope1,inter] = npy.linalg.lstsq(obsT,npy.asfarray(range(a,c)))[0]
					err1 = npy.linalg.norm(npy.asfarray(range(a,c)) - ((obsT[:,0])*slope1 + inter))
					obsT = npy.vstack([npy.asarray(tmp2[c:b]),npy.ones(len(tmp2[c:b]))]).T
					[slope2,inter] = npy.linalg.lstsq(obsT,npy.asfarray(range(c,b)))[0]
					err2 = npy.linalg.norm(npy.asfarray(range(c,b)) - ((obsT[:,0])*slope2 + inter))
					#print int(value),a,b,c,slope1,slope2,err1,err2,err1+err2
					if (err1 + err2) < err_chk:
						err_chk = (err1 + err2)
						loc = [a,b,c]
						if slope1 > 0 and slope2 < 0:
							valid = True
						else:
							valid = False
				#print "valid",valid,loc
				if valid is True:
					indx1 = npy.argmax([tmp2[m] for m in loc])
					tmp0.append(tmp2[loc[indx1]])
					tmp0a.append(loc[indx1])

				#wind1 = range(max([int(value) - int(round(options["interval"]/8.,0)),int(round(min(period),0))]),
				#  int(value) + int(round(options["interval"]/2.,0)))				# Derivative window
				tmp8.append(tmp2[int(value)])
				tmp8a.append(int(value))

				# Test if any 2nd derivatives are negative within local window - MAXIMUM in local ACF domain
				# if any(tmp2gradprime[wind1] < 0.0):
				# 	# Get max within window of point
				#
				# 	tmp0.append(npy.max(tmp2[wind1]))
				# 	tmp0a.append(tmp2.tolist().index(tmp0[-1]))
				#
				# 	#print tmp0a[-1],tmp0[-1]
				# 	tmp1.append(tmp2[int(value)])
				# 	tmp1a.append(int(value))
					
				# Trim the candidates if close to ACF maxima
				u, order = npy.unique(tmp0a, return_index=True)
				tmp0b = list(npy.asarray(tmp0a)[order])
				#tmp0b = list(tmp2[order])
				indx_trim = tmp2[tmp0b].tolist()

			#print "final",tmp0b,indx_trim
			t1,t2 = self.getRedundant(tmp0b,indx_trim,tmp2)
			tmp0c = t1
			indx_trim2 = t2
			# Trim periods to resolve and combine periods that are very close / similar
			# tmp0c = []
			# indx_trim2 = []
			# for m,val2 in enumerate(tmp0b[0:-1]):
			# 	val1 = tmp0b[m+1]
			# 	if npy.abs(val2 - val1) <= int(round(min(tmp0b)/2.,0)):
			# 		wind3 = range(min([val2,val1]),max([val2,val1]))
			# 		print wind3
			# 		indx_trim2.append(npy.max(tmp2[wind3]))
			# 		print indx_trim2
			# 		tmp0c.append(tmp2.tolist().index(indx_trim2[-1]))
			# 	else:
			# 		indx_trim2.append(indx_trim[m+1])
			# 		tmp0c.append(val1)
				# for n,val3 in enumerate(tmp0b):
				# 	if npy.abs(val3 - val2) <= int(round(min(tmp0b)/2.,0)) and val3 != val2:
				# 		wind3 = range(min([val2,val3]),max([val2,val3]))
				# 		indx_trim[n] = npy.max(tmp2[wind3])
				# 		tmp0b[n] = tmp2.tolist().index(indx_trim[n])

			# Listify output
			#tmp0c = list(set(tmp0b))
			#indx_trim2 = tmp2[tmp0c].tolist()
			#print "final2",tmp0c,indx_trim2
			# plt.plot(tmp2)
			# plt.hold(True)
			# #plt.plot(tmp2gradprime[:-5],'r')
			# plt.plot(tmp8a,tmp8,'ro')
			# plt.plot(tmp0c,indx_trim2,'ko')
			# plt.show()

			return tmp0c,indx_trim2

		except Exception as e:
			glowfish_logger.exception(e)
			raise self.exceptionClass('467.2')

	def getRedundant(self,input_indx,input_data,array1):

		"""
		Recursion to remove redundant values in periodicity array
		"""
		indx1 = [input_indx[0]]
		value1 = [input_data[0]]
		for m,val2 in enumerate(input_indx[1:]):
			if m == len(input_indx)-1:
				indx1.append(input_indx[m])
				value1.append(input_data[m])
			else:
				val1 = indx1[-1]
				#print "top",m,val1,val2
				if npy.abs(val2 - val1) <= int(round(min(input_indx)/2.,0)):
					wind3 = range(min([val2,val1]),max([val2,val1])+1)
					value2 = [npy.max(array1[wind3])]
					indx2 = [array1.tolist().index(value2[0])]

					try:
						#print "here0",value2,indx2,input_indx[m+2]
						if npy.abs(indx2 - input_indx[m+2]) > int(round(min(input_indx)/2.,0)):
							#print "continuing"
							continue
						else:
							indx2.append(input_indx[m+2])
							value2.append(input_data[m+2])

							#print "here",val1,val2,value2,indx2
							#value1.append(array1.tolist().index(indx1[-1]))
							indxN,valueN = self.getRedundant(indx2,value2,array1)
							if npy.abs(indxN[-1] - indx1[-1]) > int(round(min(input_indx)/2.,0)):
								indx1.append(indxN[-1])
								value1.append(valueN[-1])
							else:
								indx1[-1] = indxN[-1]
								value1[-1] = valueN[-1]
							#print "after",indx1,value1
					except:
						pass
				else:
					#print "here2",m,indx1,value1
					indx1.append(val2)
					value1.append(input_data[m+1])

		return indx1,value1

	def getDTWDistance(self,data_input,keylist=None):

		"""
		Compute dynamic time warping distance between all unique (upper triangular) combinations of sensors for clustering.

		:arg data_input : input hash with sensor UUID keys and time-series arrays
		:return distmat : a symmetric matrix containing the dynamic time warping distance between sensor time-series
				keylist : a well-ordered list of sensor UUIDs to be maintained from call to call for coherency of output matrix
		"""

		try:

			# Initialize output power matrix and dimensions of sensor time-series space
			dim1 = len(data_input.keys())
			distmat = npy.zeros(shape=(dim1,dim1),dtype=float)

			# Maintain keylist from call to call if it exists and index keynames for placement within distmat matrix
			if keylist is None:
				key_include = {key : None for key in data_input.keys()}
				keylist = data_input.keys()
				key_indx = {key : keylist.index(key) for key in keylist}
			else:
				key_include = {key : keylist.index(key) for key in keylist}
				key_indx = {key : keylist.index(key) for key in keylist}

			# Temporary transformation of input to numpy array for computations
			tmp = {key:npy.asarray(data_input[key]) for key in keylist}

			# Iterate over upper-triangular combinations of time-series (power distance matrix is symetric)
			for i,keyname in enumerate(keylist):

				key_include.pop(keyname)		# Exclude combinations of sensors already computed

				for j,keyname2 in enumerate(key_include.keys()):

					tmpdist = mlpy.dtw_std(tmp[keyname],tmp[keyname2],dist_only=True)
					distmat[key_indx[keyname],key_indx[keyname2]] = tmpdist
					distmat[key_indx[keyname2],key_indx[keyname]] = tmpdist

			return distmat,keylist

		except Exception as e:
			glowfish_logger.exception(e)
			raise self.exceptionClass('468.2')


	def getMetrics(self,data_in,options=None):

		"""
		Compute basic statistical metrics of input time-series.

		:arg data_in : input time-series data , options : currently unused
		:return stddev : standard deviation of input time-series
				mavg : exponential moving average of input time-series
		"""

		try:

			if options is not None and "interval" in options:
				intvl = options["interval"]
			else:
				intvl = 1

			# Compute mean of input array data
			meandat = npy.mean(npy.asarray(data_in))

			# Compute standard deviation and exponential moving average of input series
			try:
				stddev = npy.std(npy.asarray(data_in))
			except:
				stddev = 0.0

			return stddev.tolist(),meandat.tolist()

		except Exception as e:
			glowfish_logger.exception(e)
			raise self.exceptionClass('469.2')


	def getSensorClusters(self,distmat_in,keylist):

		"""
		Cluster time-series based on DTW dist matrix and flatten optimally

		:arg distmat_in : resulting DTW dist matrix returned from getDTWDistance()
				keylist : well-ordered list of sensor UUIDs passed by sensorProcess for maintaining coherency in clustering
		:return groups_out : resulting cluster group hash with format key: <group_number> : list: sensor UUIDs assigned to each group
		"""

		try:

			# Cluster and flatten to generate groups
			grouping = fclusterdata(distmat_in,0.85).tolist()

			# Entropy measure
			# for i in [0.2,0.5,0.75,0.85,1.0]:
			# 	grouping = fclusterdata(distmat_in,i).tolist()
			# 	#print entrp
			# 	print grouping

			group_lst = Counter(grouping)

			groups_out = {}
			grp_range = range(0,len(grouping))
			for keyval1 in group_lst:
				groups_out[str(keyval1)] = [keylist[p] for p in grp_range if grouping[p] == keyval1]

			return groups_out

		except Exception as e:
			glowfish_logger.exception(e)
			raise self.exceptionClass('470.2')


	@staticmethod
	def plotPOD(datainput,limits=None): #(models,hypmod=None,hypmod2=None):

		"""
		Genereric plotting method used for QA of sensorProcess class
		"""

		# Plot the results:
		fig = plt.figure()
		plt.hold(True)
		ct = 0

		color = ['r','k','b','g','y','m','r.','k.','b.','g.','y.','m.','ro','ko','bo','go','yo','mo',
				 'rx','kx','bx','gx','yx','mx']

		pltnum = len(datainput["data"].keys())

		if "time" in datainput.keys():
			time1 = npy.asarray(datainput["time"])

		if pltnum < 7:
			for keyvalue in datainput["data"]:

				model = npy.asarray(datainput["data"][keyvalue])

				plt.subplot(pltnum,1,ct)
				plt.hold(True)
				if "match" in datainput.keys():

					model2 = npy.asarray(datainput["match"][keyvalue])

					try:
						plt.plot(time1,model2,color[1],label=str('Raw Signal-'+keyvalue))
					except:
						plt.plot(model2,color[1],label=str('Raw Signal-'+keyvalue))

				if "match2" in datainput.keys():

					model3 = datainput["match2"][keyvalue]
					try:
						plt.plot(time1,model3,color[1],label=str('Raw Signal2-'+keyvalue))
					except:
						plt.plot(model3,color[1],label=str('Raw Signal2-'+keyvalue))
				# try:
				# 	#plt.plot(time1,model,color[0],label=str('New Signal-'+keyvalue))
				# 	# a_mask = model < .21
				# 	# b_mask = model >= .21
				# 	# c_mask = model >= .28
				# 	# if ct == 1:
				# 	# 	plt.plot(time1[a_mask],model[a_mask],color[9]) #,label=str('High Health-'+keyvalue))
				# 	# 	plt.plot(time1[b_mask],model[b_mask],color[10],label=str('Med Health-'+keyvalue))
				# 	# 	plt.plot(time1[c_mask],model[c_mask],color[6],label=str('Low Health-'+keyvalue))
				# 	# else:
				# 	# 	plt.plot(time1[a_mask],model[a_mask],color[9],label=str(keyvalue))
				# 	# 	plt.plot(time1[b_mask],model[b_mask],color[10])
				# 	# 	plt.plot(time1[c_mask],model[c_mask],color[6])
				#
				# except:
				plt.plot(model,color[0],label=str('New Signal-'+keyvalue))

				if "anomaly_probability" in datainput.keys():
					alt_model = npy.asarray(datainput["anomaly_probability"])
					alt_mask = alt_model > .95
					try:
						time1alt = npy.asarray(time1)
						plt.plot(time1alt[alt_mask],alt_model[alt_mask]*5.0,color[14],label="Alerts")
					except:
						plt.plot(alt_model[alt_mask]*5.0,color[14],label="Alerts")

				ct += 1
				plt.legend(loc=3)
				if limits is not None:
					plt.axis(limits)
		else:
			for keyvalue in datainput["data"]:

				model = datainput["data"][keyvalue]
				plt.subplot(6,3,ct)
				if "match" in datainput.keys():
					model2 = datainput["match"][keyvalue]
					try:
						plt.plot(time1,model2,color[1]) #,label=str('Raw Signal-'+keyvalue))
					except:
						plt.plot(model2,color[1])

				try:
					plt.plot(time1,model,color[0],label=str('Signal-'+keyvalue))
				except:
					plt.plot(model,color[0],label=str('Signal-'+keyvalue))
				plt.hold(True)

				if "anomaly_probability" in datainput.keys():
					alt_model = npy.asarray(datainput["anomaly_probability"])
					try:
						plt.plot(time1,alt_model*10.0,color[1],label="Anomaly_Probability")
					except:
						plt.plot(alt_model*10.0,color[1],label="Anomaly_Probability")

				ct += 1
				plt.legend(loc=1)
		#plt.axis([0,52000,-10,30])
		# if hypmod is not None:
		# 	plt.plot(hypmod,'b-', label='Extracted Signal 1')
		# if hypmod2 is not None:
		# 	plt.plot(hypmod2,'g-', label='Extracted Signal 2')

		#ax.set_xlim3d([xmin, xmax])
		#ax.set_ylim3d([ymin, ymax])
		#ax.set_xlabel('POD Dim 1')
		#ax.set_ylabel('POD Dim 2')

		plt.show()


if __name__ == "__main__":

	""" Class Level Tester"""

	from sys import argv,stdout
	import json
	from GlowfishException import GlowfishException
	import dateutil.parser as parser
	from scipy.stats import signaltonoise
	from datetime import datetime
	from clustering import clustering

	run_on = True

	filename = argv[1]
	file = open(filename,"r")
	keyname_1 = '573e6cab-dc75-4c77-86e5-2468831098fb'
	keyname_2 = 'fd56a7e7-ccc7-4263-9a89-05a9cc0eed6f'
	keyname_3 = 'ec2d3d39-a04a-45ac-b9e0-d755625d7166'
	#filename2 = argv[2]
	#file2 = open(filename2,"r")
	t_st = time.time()

	if run_on is True:
		try:
			data1 = json.load(file)
			print data1["data_set"]["device_17_temp"] #,data1["time_since_epoch"]
			data = {"data_set":{"device_17_temp":npy.asfarray((data1["data_set"]["device_17_temp"])).tolist()},"time":deepcopy(data1["time"])}
			print len(data["data_set"]["device_17_temp"])

			#data["data_set"].pop("time_since_epoch")
			#door = json.load(file2)

			# Format time data first
			time_sec = {}
			for keyname2 in data["data_set"]:

				#time_sec[keyname2] = data["time"]
				#print time_sec[keyname2]
				time_sec[keyname2] = [time.mktime(parser.parse(it).timetuple()) for it in data["time"]]

		except:
			raise Exception('data file is not present or decodable')

		t_en = time.time()
		print "Data loading time:",t_en - t_st

		lrner = sensorFlow(GlowfishException)

		lrner3 = pcaAnomaly(GlowfishException)

		# Stream data to sensorProcess in 4k sample windows
		sensor_list = data["data_set"].keys()
		#sensor_list = data["data_set"].keys()[0:-3]
		#sensor_list = [keyname_1,keyname_2,keyname_3]

		stpt = 12000
		endpt = 12001
		clust_on = 0
		stored_states1 = None
		data_stream = {"data":{},"time":[]}
		for i in range(stpt,endpt):
			#if i > stpt+80 and i <= stpt+150:
			#	sensor_list = data["data_set"].keys()
			#elif i > stpt+150:
			#	sensor_list = data["data_set"].keys()[0:7]
			#print "stream",i
			# Overhead handled by API and client side normally
			data_stream["data"] = {skey : data["data_set"][skey][i-min([1000,(i+1000-stpt)]):i] for skey in sensor_list}
			data_stream["time"] = {skey : time_sec[skey][i-min([1000,(i+1000-stpt)]):i] for skey in sensor_list}
			#print "Number of Sensors:",len(data_stream["data"].keys())
			###data_stream["data"] = {keyname_1:data["data_set"][keyname_1][i-min([4000,(i+1000-stpt)]):i]}
			###data_stream["time"] = {keyname_1:time_sec[keyname_1][i-min([4000,(i+1000-stpt)]):i]}
			#data_stream["data"]['8abd86de-c4a6-44ad-987e-63ee947be4ef'] = data_stream["data"]['8abd86de-c4a6-44ad-987e-63ee947be4ef'][1:-1]
			#data_stream["time"]['8abd86de-c4a6-44ad-987e-63ee947be4ef'] = data_stream["time"]['8abd86de-c4a6-44ad-987e-63ee947be4ef'][1:-1]

			t05 = time.time()		# Request time proxy

			if i > stpt:
				if clust_on >= 0:
					cluster_bool = True
					clust_on = 0
				dsp_output = lrner.sensorProcess(data_stream,cluster_bool,stored_states1)
				stored_states1 = dsp_output["states"]
				stored_states1["time_system_performance"] = dsp_output["states_time"]
				#stored_states1["time_powers"]["first"].append(float(npy.max(dsp_output["states_time"])+1.0))
				#for keyval2 in dsp_output["metrics"]["group_performance"]:
				#	print keyval2,dsp_output["metrics"]["cluster_group"][keyval2]
				ct5 = 1
				for keyval2 in dsp_output["states"]["mavg"]:
					print i,ct5,keyval2,dsp_output["states"]["primary_period"][keyval2][-1]
					print 'length input,processed',len(data_stream["data"][keyname_1]),\
						len(dsp_output["data"][keyname_1])
					ct5 += 1

				cluster_bool = False
				clust_on += 1
			else:

				cluster_bool = True
				dsp_output = lrner.sensorProcess(data_stream,cluster_bool)
				stored_states1 = dsp_output["states"]
				stored_states1["time_system_performance"] = dsp_output["states_time"]
				#stored_states1["time_powers"] = {"first":[float(npy.min(dsp_output["states_time"])-1.0)]}
				#stored_states1["time_powers"]["first"].append(float(npy.max(dsp_output["states_time"])+1.0))
				cluster_bool = False
				clust_on += 1

		dsp_new = deepcopy(dsp_output)
		#print dsp_new["states"].keys()
		print "writing output"
		file5 = open("stats.json","w")
		json.dump(dsp_new,file5)
		file5.close()

	else:
		lrner = sensorFlow(GlowfishException)
		file5 = open("stats.json","r")
		dsp_new = json.load(file5)
		file5.close()

	timetemp = []
	for it in dsp_new["processed_time"]:
		timetemp.append(datetime.fromtimestamp(int(it)))
	dsp_new["time"] = timetemp

	# Plot results
	if dsp_new["return"] is True:
		groupnew = {}
		timenew = {}
		match = {}
		match2 = {}
		#for groupname in dsp_new["metrics"]["cluster_group"]:
		#	groupnew[groupname] = {"data":{},"match":{}}
		for device in dsp_new["data"]:
			groupnew[device] = dsp_new["states"]["sensor_performance"][device][:]
			timenew = dsp_new["processed_time"]
			match[device] = data["data_set"][device][0:i]
			match2[device] = dsp_new["states"]["primary_period"][device][:]
			#match2[device] = dsp_new["states"]["primary_period"][device][:]
			print npy.mean(match[device]),i
			print 'length output',len(groupnew[device]),len(match2[device]),len(timenew)
		lrner.plotPOD({"data":groupnew,"match":match,"match2":match2})