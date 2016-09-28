#Signal Processing Library

Time-Series signal processing class which accomodates any number of time series contained under subkeys of the input
dictionary. Time series do not have to be the same length or sampled at the same frequencies.

####Implemented Methods####

replaceNullData - method for simple replacement of a single known amplitude from input series (e.g., known error codes)

despikeSeries - method for adaptive despiking with minimal distortion using Savitsky-Golay filtering, Otsu's method for
                thresholding, and interpolation for spike data replacement. Threshold seeks to maximize inter-class
                variance between "spike" class and "normal" class.

registerTime - method for multiple time series time registration using linear interpolation. Interpolation made to
               mean sampling rate of n-dim sensor time series with outliers removed before mean sample rate is computed.

getPrimaryPeriods - method for automatically picking primary periods from multiple input series based on
                collaboration between periodogram and auto-correlation function to tolerate long + short periodicities
                (See Vlachos, Yu, & Castelli, 2005). Also returns SNR for each resulting primary periods.

####input####
    data_input (dictionary) - contains nested dictionaries with two high level keys: {"data" : {}, "time" :{}}
    "data" : dictionary containing key:data series pairs (lists) for each time-series to be processed (of potentially different lengths)
    (optional) "time" : dictionary containing key:timestamp series pairs (lists) corresponding to timestamps for each datum in "data"
    (optional) "options" (dictionary) - contains options for various methods as follows:

    replaceNullData() - options["value"] (float) - single value to be replaced in all time-series input data
    despikeSeries() - options["window"] (odd-integer) - single integer value to be used for local despike window
    registerTime() - options["sample"] (integer) - single factor for downsampling output (2 == 1/2 sampling of optimal sampling)

####notes####
    1) sub key names under top-level key "data" must correspond to same key names under top-level key "time"
    2) if time-series lengths are different, registerTime() can be run before other methods to produce equal sampling
    3) timestamps (if included) can be of type :
            datetime strings (e.g., '2015-12-09T20:33:04Z')
            ms since epoch as floats (e.g., 1449257205.0)

            method validateTime() is run on class instantiation to convert any datetime strings to ms since epoch floats

####return####
    dictionary containing nested dictionaries with two high level keys: {"data" : {}, "time" :{} }
    "data" is dictionary containing key : data pairs (list) of resulting processed time-series data
    if registerTime() is run :
        "time" : list of resulting resampled time shared for all input time series in "data" (ms since epoch format)
    else :
        "time" input key "time" is passed to output dictionary key "time" in milliseconds since epoch format

####Usage####

See /tests/signal_process_tests.py for example usage for methods implemented here

####dependencies####
    scipy

####References####
    Vlachos, Yu, & Castelli, 2005, "On Periodicity Detection and Structural Periodic Similarity"
