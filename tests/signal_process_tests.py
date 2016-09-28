#!/usr/bin/env python2.7
# encoding: utf-8

"""
Some unit tests and usage examples for signal processing library

@author Michael Tompkins
@copyright 2016__. All rights reserved.
"""

if __name__ == "__main__":

	"""
	Unit Tests

	Run tests for time key or no time key for all routines
	Run tests for timeRegister of multiple series input
	Show chaining of methods for despike with sensor13 given min periodicity
	Run tests for all options
	Run replacement test
	Generate plots for some outputs for documentation
	"""

	import numpy as npy
	from datetime import datetime
	import json
	import time
	import matplotlib.pyplot as plt

# TODO change test data filenames and relative paths
# TODO Move unit tests to /tests directory
# TODO Move datetimes around in test file to get them to overlap for the most part - then run registerTime tests
# TODO Write readme file with tests and some plots

	run_period = True
	run_time_register = True
	run_depike = False
	run_replace = False

	print "Loading some time-series data\n"
	t_st = time.time()

	filename = "/workspace/Sampling-Integration-Interpolation/tests/three_time_series_data.json"
	data_in = {"data":{},"time":{}}
	file1 = open(filename,"r")
	data1 = json.load(file1)
	for name in data1["data"]:
		data_in["data"][name] = npy.asfarray(data1["data"][name]).tolist()
		data_in["time"][name] = data1["time"][name]

	print "Time-Series Labels: ",data_in["data"].keys()

	t_en = time.time()
	print "Data Load Time:",t_en - t_st,"\n"

	if run_period is True:

		t_st = time.time()
		options = None
		lrner = signal_processing.signalProcess(data_in,options)
		output = lrner.getPrimaryPeriods()
		t_en = time.time()
		print "Processing Time: ",t_en - t_st," secs\n"
		print output

	if run_time_register is True:
		t_st = time.time()
		options = {"sample":1}
		lrner = signalProcess(data_in,options)
		data_out, params_out = lrner.registerTime()
		t_en = time.time()
		print "Processing Time: ",t_en - t_st," secs\n"
		print data_in["data"].keys()
		# plot before and after

		datetime_vals = [datetime.fromtimestamp(int(it)) for it in data_out["time"]]
		for key in data_out["data"]:
			plt.hold(True)
			plt.plot(datetime_vals,data_out["data"][key],label=key)
			plt.legend()
		plt.show()

	# timetemp = []
	# for it in dsp_new["processed_time"]:
	# 	timetemp.append(datetime.fromtimestamp(int(it)))
	# dsp_new["time"] = timetemp