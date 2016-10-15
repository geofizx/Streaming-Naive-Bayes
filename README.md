# Streaming Naive Bayes Classifier

####Description

	Streamning Naive Bayes Classifier including dynamic re-binning, training, predictions, and classification
	error statistics. Handles heterogeneous feature sets with mixed INT, FLOAT, STRING feature types. Also handles
	missing feature values explicitly.

	Includes dynamic binning to account for updated feature dynamic ranges over data stream

####Notes
	re-binning implemented here is quite crude and needs to be improved

####Methods Implemented
    classLearn - Training using continuous (float), categorical (int or string), and mixed feature data types
	classPredict - Predictions based on Naive Bayes Classifier
	classUpdate - Update method for incremental (streaming) training
	classAcc - Classification error statistics method
	validateFeatures - Feature data type validation and mapping for learner

####Input
    features dictionary with <feature_name> : [values] (list) pairs for every feature
    target class dictionary in form "class" : [target class values] (list of ints or strings)

####Dependencies####
    collections.Counter

####Usage
    /tests/bayes_tests.py - example usage and tests for known 2-class classification problem in UCI Bank Train data set.

#### License ####
* [MIT License](LICENSE.md)

