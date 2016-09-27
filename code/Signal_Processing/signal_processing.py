#!/usr/bin/env python2.7
# encoding: utf-8

"""

@description

	Time-Series signal processing class which handles multiple time-series from input dictionary.
	The following methods are implemented :

	replaceNullData - method for simple replacement of a single known amplitude from input (e.g., known error codes)

	despikeSeries - method for adaptive input despiking using Savitsky-Golay filtering, Otsu's method for
					thresholding, and interpolation for spike data replacement.
					Threshold seeks to maximize inter-class variance between "spike" class and "normal" class

	registerTime - method for multiple time-series time registration using linear interpolation. Interpolation made to
					mean sampling rate of n-dim sensor time-series with outliers removed before mean sample rate is computed.

	getPrimaryPeriods - method for automatically picking primary periodicity from multiple input series based on
					collaboration between periodogram and auto-correlation function to tolerate long + short periodicities


@input : data_input (dictionary) - contains nested dictionaries with two high level keys: {"data" : {}, "time" :{} }
		"data" is dictionary containing key : data pairs (list) for each time-series to be processed (of potentially different lengths)
		"time" is dictionary containing key : timestampe pairs (list) corresponding to timestamps for each list in "data"
		options (dictionary) - contains options for various methods as follows:

		replaceNullData() - options["value"] (float) - single value to be replaced in all time-series input data
		despikeSeries() - options["window"] (odd-integer) - single integer value to be used for local despike window
		registerTime() - options["sample"] (integer) - single factor for downsampling output (2 == 1/2 sampling of optimal sampling)

@notes :
		1) sub key names under top-level key "data" must correspond to same key names under top-level key "time"
		2) if time-series lengths are different, than registerTime() must be run first before other methods
		3) if time-series are same length, then any other method can be run independently without running registerTime()

@return : dictionary containing nested dictionaries with two high level keys: {"data" : {}, "time" :{} }
		"data" is dictionary containing key : data pairs (list) of resulting processed time-series data
		if registerTime() is run :
			"time" is dictionary containing sub key "time" with resulting resampled time for all input time series in "data"
		else :
			"time" input key "time" is passed to output dictionary key "time"

@author michael@glowfish.io
@date 2016-02-10
@copyright (c) 2016__. All rights reserved.
"""

# TODO - handle presence of "time" key in input - exception raised if timeRegister calles and no time present
# TODO - handle timestamp format conversion in init method and replace input "time" key by this

# TODO test case when options are not defined and when they are
# TODO check for odd/even despike windows input
# TODO test for input window versus default for unit test
# TODO test all methods
# TODO Unit tests

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

npy.seterr(all='raise')
import logging
glowfish_logger = logging.getLogger('fisherman.lineSinker')

class signalProcess:

	def __init__(self,data_input,options=None):

		# Initialize class varialbles
		self.debug = True
		self.small = 1.0e-10
		self.large = 1.0e+32
		self.neg = -1.0e+10
		self.window = None
		self.value = None
		self.sample = 1

		"""
		Interface for signalProcess methods

		If options=None, despikeSeries() will be performed with default window (5 samples), and replaceNullData() will
		return error.

		else, if options is supplied by input, a dictionary containing the following options is supported:
		key - window : odd integer required
		key - value : float value required

		:arg data_input input sensor time-series raw data dictionary with high-level keys "data" and "time". "data"
					has identifier for each series that corresponds to same key names under key "time" containing the
					timestamps for each series under key "data"
		"""

		# Check for presence of processing options
		if options is not None:
			if 'window'in options:
				self.window = options['window']
			if 'value' in options:
				self.value = options['value']
			if 'sample' in options:
				self.sample = options['sample']

		self.data = data_input


	def replaceNullData(self):

		"""
		N-dim data series replacement of single self.value float data from series by linear interpolation.

		:arg self.data : dictionary with key "data" - dictionary with key : time-series raw data float lists
						and optionally key "time" (optional) with key : time-series timestamp lists
		:return datanew : dictionary with key "data - dictionary with key time-series after replacement of self.value
		"""

		try:

			dataN = deepcopy(self.data)

			# Handle case with and without high-level key "time"
			if "time" in self.data:
				datanew = {'data':{},'time':self.data["time"]}
			else:
				datanew = {'data':{}}

			# Index any samples == self.value for each time-series in input dictionary
			for m,keys in enumerate(dataN['data']):
				samples = range(0,len(dataN['data'][keys]))
				times = npy.asfarray(dataN['time'])
				tmp_array = npy.zeros(shape=(len(samples)),dtype=float)
				inx_null = [g for g in samples if dataN['data'][keys][g] - (self.value) <= self.small]
				inx_ok = [g for g in samples if dataN['data'][keys][g] - (self.value) > 1]

				if len(inx_null) == 0:	# No replaced values in returned series
					tmp_array[:] = dataN['data'][keys]

				elif len(inx_ok) == 1:	# Only 1 good value in input series
					tmp_array[:] = dataN['data'][keys][inx_ok[0]]

				elif len(inx_ok) == 0:	# No GOOD values in array so maintain all input data in returned series
					tmp_array[:] = self.value

				else:					# At least 2 good values in input series to interpolate

					# Handle case of first point to set lower bound for interpolation
					if inx_null[0] == 0:
						dataN['data'][keys][0] = dataN['data'][keys][inx_ok[0]]

					# Handle case of last point to set upper bound for interpolation
					if inx_null[-1] == len(dataN['data'][keys])-1:
						dataN['data'][keys][-1] = dataN['data'][keys][inx_ok[-1]]

					# Re-index replaced values with lower/upper bounds set from above
					inx_mod = [g for g in samples if dataN['data'][keys][g] - (self.value) > 1]

					# Handle interior points with cubic spline interpolant
					signal_intrp = interp1d(times[inx_mod],npy.asfarray(dataN['data'][keys])[inx_mod])
					tmp_array = signal_intrp(times)

				datanew['data'][keys] = tmp_array.tolist()

			# Compute average sampling rate for all times
			#device_delta_t = float(npy.mean(npy.asfarray(datanew['time'][1:]) - npy.asfarray(datanew['time'][0:-1])))

			return datanew

		except Exception as e:
			raise Exception (e)


	def despikeSeries(self):

		"""
		Data series despiking using Savitsky-Golay filtering, Otsu's method for thresholding, and
		interpolation for spike data replacement

		:arg self.data : dictionary : multiple time-series lists under key "data", with corresponding key names under
						key "time" with lists of monotonic timestamps sampling for each time-series in key "data"
		:arg (optional) : self.window : integer : sample length of local despike window (must be odd integer)
		:return signal_out : dictionary with same keys "data" and "time" of same size as input dictionary but with
							processed data returned under dictionary "data"

		Exceptions Handled : If any time-series has zero variance (equal values), then input time-series list is returned
		"""

		# Handle case with and without high-level key "time"
		if "time" in self.data:
			signal_out = {'data':{},'time':self.data["time"]}
		else:
			signal_out = {'data':{}}

		# Define local odd-integer despike window or use default window length = 5
		if self.window is not None:
			# Check if odd integer, else translate to nearest odd integer < window_opt
			if self.window/2.0 == self.window/2:
				winlen = self.window - 1
			else:
				winlen = self.window
		else:
			winlen = 5	# must be an odd integer of order of expected spike sample width x (1.5 - 3)

		# Pad array by 1 sample to ensure that spike at last sample is handled properly
		padlen = 1

		# Despike all series in input dictionary
		for data_key in self.data["data"]:

			try:		# Exception handling for any single time-series containing same value for entire list

				# Copy of input signal for spike replacement
				sigint2 = npy.asfarray(self.data["data"][data_key])

				# Pad the signal by mean of previous 20-sample signal above global time-series mean value to
				# avoid poor interpolation at the endpoint of array for streaming time-series
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

				#Handle 'small-amplitude' case where NO spike is greater than 5% of interpolated signal then ignore threshold
				if npy.max(diffperc) < .05:

					signalout = sigint			# Return raw data

				else:		# Return despiked signal

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

						# Handle case of consecutive points belonging to the same spike
						if x[k+1]-x[k] <= 1:
							k += 1

						# Handle case of separated spikes
						else:
							stopspike = x[k]
							spikes[i] = npy.floor((stopspike+startspike)/2.)
							startspike = x[k+1]

							# Step 2: excision of the spike region, s, of width Wspike (centered on
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

					# Final replacement of spike regions for return
					for i in range(0,samples):
						s = range(jmin[i],jmax[i])
						signalout[s] = tmpsig[s]

				# Pack return dictionary
				signal_out["data"][data_key] = signalout[0:-padlen].tolist()

			except:

				# Pack same input key series to return dictionary
				signal_out["data"][data_key] = self.data["data"][data_key]

		return signal_out


	def registerTime(self):

		"""
		N-dim data series time registration using linear interpolation. Interpolation made to mean
		sampling rate of n-dim sensor time-series with outliers removed before mean sample rate is computed.

		:arg self.data : dictionary : multiple time-series lists under key "data", with corresponding key names under
						key "time" with lists of monotonic timestamps sampling for each time-series in key "data"
		:arg (optional) : self.sample : integer : downsampling factor (default == 1 - no downsampling, 2== 1/2 sampling)
		:return signal_out : dictionary with same keys "data" and "time" of same size as input dictionary but with
							processed data returned under dictionary "data"

		:return datanew : interpolated time-series dictionary key "data" with additional key "time" representing the new
				timestamp list representing shared time samples for all input time-series keys
				device_min : minimum of device sample rates
				device_max : maximum of device sample rates
				device_delta_t : the resulting mean time delta (sampling rate) for all sensor time-series returned
		"""

		try:

			sample_rate = self.sample
			dataN = deepcopy(self.data)

			if "time" not in self.data:
				raise Exception("'time' key with timestamp data must be present for method registerTime")

			datanew = {'data':{},'time':{}}
			dims2 = len(self.data['data'].keys())

			key_order = []
			min_sample = 0.0
			min_start = self.large
			max_start = 0.0

			# Determine high/low time range and lowest sampling rate to perform registration over all dimensions of input
			# Throw away extreme sample rates 95th percentile for sample_rate computation only
			# Throw away interpolated data outside range of all feeds individually
			mask = {}
			device_min = {}
			device_max = {}

			# Determine mean sampling for all time series in input
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
				min_sample = npy.max([min_sample,npy.mean(all_samples[mask[keys]])])

			# Down sampling if requested by update_dict["sample_rate"]
			min_sample *= float(sample_rate)
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
			raise Exception (e)


	def getAutocorrelation(self):

		"""
		Compute auto-correlation function for each 1-D series (lists) under key self.data["data"] used in conjunction
		with power spectrum to solve for optimal periodicity in time-series window

		:arg self.data : input dictionary with keys for each input time-series list
		:return acf_out : dictionary with autocorrelation coefficient lists of length of input array for each input key
		"""

		try:
			acf_out = {}
			for data_key in self.data["data"]:
				n = len(self.data["data"][data_key])
				#nlags = n/10
				#variance = data_in.var()
				x = self.data["data"][data_key] - self.data["data"][data_key].mean()
				r = npy.correlate(x, x, mode = 'full')[-n:]
				acfout = r/npy.max(r)

				acf_out[data_key] = acfout.list()

			return acf_out

		except Exception as e:
			raise Exception (e)


	def getPeriodogram(self,series_in,delta_t,options=None):

		"""
		Compute periodogram of discrete function and return signal to noise, powers, and dominant periods. Used in
		conjunction with ACF and gradient methods to optimize signal periodicity.

		:arg self.data : input time-series dictionary with key : data (list) for each time-series to evaluate
		:return periods_out : dictionary with high-level keys :
				"periods" : array of only the dominant periods with signal above 95% noise percentile
				"pxx_den" : array of only the dominant powers (amplitudes) with signal above 95% noise percentile
				"s_noise_l" : estimated SNR for input time-series dominant periods
		"""

		try:

			periods_out = {"periods":{},"pxx_den":{},"s_noise_l":{}}

			for data_key in self.data["data"]:

				series_in = deepcopy(self.data["data"][data_key])

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

				periods_out["periods"] = npy.asarray(periods)[Pmask][Pmaskf]
				periods_out["pxx_den"] = Pxx_den[Pmask][Pmaskf]
				periods_out["s_noise_l"] = s_noise_l

		except Exception as e:
			raise Exception (e)

		#return npy.asarray(periods),Pxx_den,npy.asarray(periods)[Pmask][Pmaskf],Pxx_den[Pmask][Pmaskf],s_noise_l
		return periods_out


	def getPowerDistanceAndSNR(self,tmp0a,periods_indx,periods_old,fullperiods,fullpower,snrL):
		"""
		Compute power distance between previous and current periodicity of discrete function by interpolation of current
		periodogram series to ACF time periods and computing L2 norm of previous states - current states.
		Also convert input SNR to units: dB.

		:arg tmp0a (list of dominant periods), periods_indx (ist of indexes of dominant periods within entire power series)
				periods_old (previous periods state computed),fullperiods (full array of periods),fullpower (full array of powers)
				snrL (input SNR ratio computed from getPeriodogram())
		:return power_new (computed power distance from previous to current state)
				periods_new (current dominant periodocity list)
				snr_new : current SNR in dB
		"""

		try:

			#Exceptions if no peak period present in input periods list or begining from first call in time-series
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
			raise Exception (e)


	def getPrimaryPeriods(self,period,tmp2):

		"""
		Map candidate dominant time-series periods returned by getPeriodogram() to ACF time lags, then determine by first
		and second derivatives of ACF whether these candidates map to maxima within a local window of ACF series, if so,
		segment window between +- 1/10 * period to estimate nearest maximum and assign new periods to these ACF maxima.

		Method : "On Periodicity Detection and Structural Periodic Similarity", Vlachos, Yu, & Castelli, 2005

		:arg self.data
		:return periodicity : dictionary containing keys :

				"periods" : dictionary containing final estimated periods list for each input time-series key in self.data["data"]
				"tmp0c" : dictionary containing indexes (from complete time series) of final periods lists from key "periods"
		"""

		try:

			tmp0 = []
			tmp0a = []

			# Iterate over all input dominant periods returned by getPeriodogram() and input here
			tmp8 = []
			tmp8a = []
			for value in period:

				# Set local ACF search window for determining if derivatives are maximal locally
				wind2 = range(max([int(value) - int(round(min(period)/2.,0)),int(round(min(period)/2.,0))]),
				  max([int(value) + int(round(value/20.,0)),int(value) + int(round(min(period)/2.,0))]))

				err_chk = 1.0e+10
				valid = False
				loc = None
				for k in range(min([18,len(wind2)-2])):	# Perform 10-bisection regression to get optimal split of window
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
			plt.plot(tmp2)
			plt.hold(True)
			#plt.plot(tmp2gradprime[:-5],'r')
			plt.plot(tmp8a,tmp8,'ro')
			plt.plot(tmp0c,indx_trim2,'ko')
			plt.show()

			return tmp0c,indx_trim2

		except Exception as e:
			raise Exception (e)


	def getRedundant(self,input_indx,input_data,array1):

		"""
		Recursion to remove nearly redundant values in periodicity array
		Note: needs some refactoring
		"""

		indx1 = [input_indx[0]]
		value1 = [input_data[0]]
		for m,val2 in enumerate(input_indx[1:]):
			if m == len(input_indx)-1:
				indx1.append(input_indx[m])
				value1.append(input_data[m])
			else:
				val1 = indx1[-1]
				if npy.abs(val2 - val1) <= int(round(min(input_indx)/2.,0)):
					wind3 = range(min([val2,val1]),max([val2,val1])+1)
					value2 = [npy.max(array1[wind3])]
					indx2 = [array1.tolist().index(value2[0])]

					try:
						if npy.abs(indx2 - input_indx[m+2]) > int(round(min(input_indx)/2.,0)):
							continue
						else:
							indx2.append(input_indx[m+2])
							value2.append(input_data[m+2])
							indxN,valueN = self.getRedundant(indx2,value2,array1)
							if npy.abs(indxN[-1] - indx1[-1]) > int(round(min(input_indx)/2.,0)):
								indx1.append(indxN[-1])
								value1.append(valueN[-1])
							else:
								indx1[-1] = indxN[-1]
								value1[-1] = valueN[-1]
					except:
						pass
				else:
					indx1.append(val2)
					value1.append(input_data[m+1])

		return indx1,value1


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

	"""
	Class Level Tests
	"""

	from sys import argv
	import json
	import dateutil.parser as parser
	from datetime import datetime

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

		options = None
		lrner = signalProcess(data,options)

		# Stream data to sensorProcess in 4k sample windows
		sensor_list = data["data_set"].keys()
		#sensor_list = data["data_set"].keys()[0:-3]
		#sensor_list = [keyname_1,keyname_2,keyname_3]

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