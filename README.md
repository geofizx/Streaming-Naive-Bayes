# Streaming Naive Bayes Classifier

####Description

	Streaming Naive Bayes Classifier including dynamic re-binning, training, predictions, and classification
	error statistics. Handles heterogeneous feature sets with mixed INT, FLOAT, STRING feature types. Also handles
	missing feature values explicitly.

####Notes
	Includes dynamic binning to account for updated feature dynamic ranges over data stream, re-binning implemented here
	is quite crude and needs to be improved.

####Methods Implemented
    class_learn - Training using continuous (float), categorical (int or string), and mixed feature data types
	class_predict - Predictions based on Naive Bayes Classifier
	class_update - Update method for incremental (streaming) training
	class_acc - Classification error statistics method
	validate_features - Feature data type validation and mapping for learner

####Input
    features dictionary with <feature_name> : [values] (list) pairs for every feature
    target class dictionary in form "class" : [target class values] (list of ints or strings)

####Python Versions Supported####
    Python 2.7.x

####Usage
    /tests/bayes_tests.py - example usage and tests for known 2-class classification problem in UCI Bank Train data set.

####Unit Tests
    /code/test_classifier.py

####Test Data
    UCI dataset: [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of
    Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

#### License ####
* [MIT License](LICENSE.md)

