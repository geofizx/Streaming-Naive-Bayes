#!/usr/bin/env python2.7
# encoding: utf-8

"""
Some example training/prediction/error and usage examples
for streaming Naive Bayes classifier

Data - UCI Data:
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita.
A Data-Driven Approach to Predict the Success of Bank Telemarketing.
Decision Support Systems, Elsevier, 62:22-31, June 2014

Author Michael Tompkins
Copyright 2016
"""

import json
from code import Classifier

# Determine which tests are run below
batch = True
stream = True

data_train = "bank_train_data.json"
data_test = "bank_test_data.json"

# Load feature and class training and testing data from file
dfile = open(data_train, "r")
data1 = json.load(dfile)
train_responses = data1['response']
data1.pop("response")
train_data = data1["data_set"]

dfile = open(data_test, "r")
data2 = json.load(dfile)
test_responses = data2['response']
data2.pop("response")
test_data = data2["data_set"]

if batch is True:		# Learn in one batch call to classifier

    # Instantiate and train classifier
    lrner = Classifier()
    [priors_dict, pars_dict] = lrner.class_learn(train_data,
                                                 train_responses)

    # Make some batch predictions
    class_dict2 = lrner.class_predict(priors_dict,
                                      pars_dict,test_data)

    # Compute prediction statistics
    stats = lrner.class_acc(class_dict2,test_responses,"all")
    print "Batch Learning Error Statistics"
    for l,classname in enumerate(stats["class_names"]):
        print "Predicted Class : ", classname," Recall : ",\
            stats["recall"][l], " Precision :", stats["precision"][l], \
            " F1 Score :", stats["f1_scores"][l],\
            " Composite Accuracy :", stats["Composite_Accuracy"]

if stream is True:  # Learn sequentially over data stream

    # Instantiate and train classifier
    lrner = Classifier()

    # Learn incrementally by streaming training data 2 points at a time
    priors_dict = {}
    pars_dict = {}
    tot = len(train_responses["class"])
    for k in range(0,tot,2):
        datatmp = {m : train_data[m][k:k+2] for m in train_data.keys()}
        responsetmp = {m : train_responses[m][k:k+2] for m in train_responses.keys()}
        if k == 0:
            # Train on first two instances
            [priors_dict, pars_dict] = lrner.class_learn(datatmp,
                                                         responsetmp)
        else:
            # Update sequentially
            [priors_dict, pars_dict] = lrner.class_update(priors_dict, pars_dict,
                                                          datatmp,responsetmp)

    # Make some batch predictions
    class_dict2 = lrner.class_predict(priors_dict,pars_dict,test_data)

    # Compute prediction statistics
    stats = lrner.class_acc(class_dict2,test_responses,"all")
    print "Streaming Learning Error Statistics"
    for l, classname in enumerate(stats["class_names"]):
        print "Predicted Class : ", classname, " Recall : ", stats["recall"][l], \
            " Precision :", stats["precision"][l], \
            " F1 Score :", stats["f1_scores"][l],\
            " Composite Accuracy :", stats["Composite_Accuracy"]