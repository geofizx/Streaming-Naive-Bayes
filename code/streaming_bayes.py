#!/usr/bin/env python2.7
# encoding: utf-8

"""
Desciption

    Streaming Naive Bayes Classifier including dynamic re-binning,
    training, predictions, and classification statistics. Handles
    heterogeneous feature sets with mixed INT, FLOAT, STRING
    feature types. Also handles missing values from feature set.

    Methods Implemented :
    1. class_learn() - Training using continuous (float),
        categorical (int or string), and mixed feature data types
    2. class_predict() - Predictions based on Naive Bayes Classifier
    3. class_update() - Incremental (streaming) training method
    4. class_acc() - Classification error statistics method
    5. validate_features() - Feature data type validation/mapping

Notes
    Includes dynamic binning to account for updated feature dynamic
    ranges over data stream, but re-binning implemented here is quite
    crude and needs to be improved.

Author
Michael Tompkins
Copyright 2016
"""

# Externals
from collections import Counter
import copy
import numpy as npy

class Classifier:

    """
    Streaming Naive Bayes Classifier
    """

    def __init__(self):

        self.listmap = None
        self.debug = True
        self.small = 1.0e-10

    def validate_features(self, feature_set):

        """
        Validation method for data types in feature space

        Args:
        :param feature_set: (dict) <feature_name> : [feature list]

        Returns:
        :return: listmap: (dict) <feature_name>: <data type> pairs
                for all features in space
        """

        self.listmap = {}

        for keyname in feature_set:

            if any(isinstance(n, float)
                   for n in feature_set[keyname][:]):
                fvalue = float
            elif any(isinstance(n, int)
                     for n in feature_set[keyname][:]):
                fvalue = int
            elif any(isinstance(n, (str, unicode))
                     for n in feature_set[keyname][:]):
                fvalue = str
            else:
                raise Exception(
                    "Input features must be type int, float, or string")

            self.listmap[keyname] = fvalue

        if self.debug is True:
            print self.listmap

        return

    def class_learn(self, train, response):

        """
        Naive Bayesian Classifier Initial Training Method
        Train and build frequency table from initial training
        feature set data stream.

        Args:
        :param train: (dict) <feature name>:[feature values] pairs
        :param response: (dict) <target class name>:[target class values]

        Returns:
        :return freqs: (list of dicts) updated priors/likelihoods
        :return params: (dict) updated feature parameter states
        """

        try:

            # Validate feature vector data types on initial
            # learning and generate self.listmap dictionary used here
            self.validate_features(train)

            # Row Count and Sanity Check for Bayes Classifier
            rowtmp = []
            for keyname in train:
                rowtmp.append(len(train[keyname]))  # Row count

            rowcount = rowtmp[0]  # Assign count to first index
            numatt = len(train)   # Number of attributes

            # Validate there is only a single key in classes dictionary
            keyval = response.keys()
            if len(keyval) > 1:
                raise Exception("more than 1 key for target data")

            # Enumerate the classes present in data
            classlist = response[keyval[0]]
            count_up = Counter(classlist)
            classes = count_up.keys()
            numclass = len(classes)
            class_ct = [count_up[m] for m in classes]

            if self.debug:
                print class_ct
                print 'rowcount, numatt, numclass'
                print rowcount, numatt, numclass
                print classes
                print numclass

            # Determine number of unique strings in each
            # feature vector of data type str
            strings = {}
            str_range = {}
            for key1 in train:
                if isinstance("str", self.listmap[key1]):
                    strings[key1] = Counter(train[key1]).keys()
                    str_range[key1] = len(strings[key1])

            # Set up frequency tables for learning
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

            freqs = []
            indexes = []
            for m in range(0, numclass):
                indexes.append([])
                classname = classes[m]
                freqs.append({})
                indexes[m] = [i for i, x in enumerate(classlist)
                              if x == classname]

            # Build frequency table one feature vector
            # at a time and handle missing values
            for key in train:

                if self.listmap[key] is int:

                    tmplist = [i for i in train[key][:]
                               if isinstance(i, int)]
                    bin_min[key] = len(npy.unique(tmplist))
                    int_range[key] = npy.max(tmplist) - npy.min(tmplist)
                    int_min[key] = npy.min(tmplist)
                    int_max[key] = npy.max(tmplist)
                    tmpl = int_range[key] / 6
                    bin_num = npy.max([tmpl, 1])
                    # bin_tmp = int_range[key] / bin_num + 1
                    bin_inc[key] = [bin_num + 1 for i in range(0, 6)]
                    int_bins[key] = [int_min[key]]

                    for g in range(0, len(bin_inc[key])):
                        int_bins[key].append(int_bins[key][g] +
                                             bin_inc[key][g])

                    # Enumerate classes for feature
                    for j, classname in enumerate(classes):
                        t = [train[key][i] for i in indexes[j]
                             if isinstance(train[key][i], int)]
                        if j == 0:
                            bins = int_bins[key]
                            [values, bins] = \
                                npy.histogram(t,
                                              bins,
                                              (int_min[key],
                                               int_max[key]))
                        else:
                            [values, bins] = \
                                npy.histogram(t,
                                              bins,
                                              (int_min[key],
                                               int_max[key]))

                        freqs[j][key] = list(values)

                elif self.listmap[key] is float:

                    # Handle NaN missing values
                    tmplist = [i for i in train[key][:] if not npy.isnan(i)]
                    if len(tmplist) == 0:
                        float_range[key] = 0.0
                        float_min[key] = 0.0
                        float_max[key] = 0.0
                        float_bins[key] = [0.0]
                        float_inc[key] = [0.0]

                    else:
                        float_range[key] = npy.ceil(npy.max(tmplist) -
                                                    npy.min(tmplist))
                        float_min[key] = npy.min(tmplist)
                        float_max[key] = npy.max(tmplist)
                        [values, binsf] = npy.histogram(tmplist, 6)
                        float_bins[key] = list(binsf)
                        float_inc[key] = []
                        for g in range(0, len(float_bins[key]) - 1):
                            float_inc[key].append(float_bins[key][g + 1] -
                                                  float_bins[key][g])

                    # Enumerate classes for feature
                    for j, classname in enumerate(classes):
                        t = [train[key][i] for i in indexes[j]
                             if isinstance(train[key][i], float)]
                        if j == 0:
                            bins = float_bins[key]
                            [values, bins] = \
                                npy.histogram(t, bins,
                                              (float_min[key],
                                               float_max[key]))
                        else:
                            [values, bins] = \
                                npy.histogram(t, bins,
                                              (float_min[key],
                                               float_max[key]))

                        freqs[j][key] = list(values)

                # Str features
                elif isinstance("str", self.listmap[key]):

                    # Enumerate over classes
                    for j, classname in enumerate(classes):
                        freqs[j][key] = \
                            [0 for n in range(0, str_range[key])]
                        t = [train[key][i] for i in indexes[j]]
                        str_cntr = Counter(t)
                        for k, stringkey in enumerate(strings[key]):
                            freqs[j][key][k] = str_cntr[stringkey]

                else:
                    raise Exception("Unprocessable Feature "
                                    "Type (int,float,string supported)")

        except Exception as e:
            raise Exception(e)

        params = {'int_range': int_range, 'bin_inc': bin_inc,
                  'int_bins': int_bins, 'int_min': int_min,
                  'float_range': float_range, 'float_inc': float_inc,
                  'float_bins': float_bins, 'float_min': float_min,
                  'numatt': numatt, 'numclass': numclass,
                  'classes': classes, 'class_ct': class_ct,
                  'strings': strings, 'int_max': int_max,
                  'float_max': float_max}

        return freqs, params

    def class_update(self, freqs, params, update, response):

        """
        Naive Bayesian Classifier Update Method
        Incrementally augment priors and frequency tables
        (and potentially bins) with attributes of [update]
        set from previous training

        Args:
        :param freqs: (list of dicts) priors/likelihoods from prev streams
        :param params: (dict) feature parameter states from prev streams
        :param update: (dict) current feature space stream
                        as <feature_name> : <feature_value_list>
        :param response: (dict) current target classes for
                        current feature space stream

        Returns:
        :return freqs: (list of dicts) updated priors/likelihoods
        :return params: (dict) updated feature parameter states
        """

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
        numclass = params['numclass']
        classes = params['classes']
        strings = params['strings']

        # Check feature label integrity between calls to this method
        if len(update.keys()) == len(freqs[0].keys()):
            for key1 in update:
                if key1 not in freqs[0].keys():
                    raise Exception("Feature label mismatch. "
                                    "Specify new set if needed.")
        else:
            raise Exception("Feature label mismatch. "
                            "Set reset_model to true or "
                            "specify new set if needed.")

        try:

            length = len(update[update.keys()[0]])
            keyval = response.keys()
            if len(keyval) > 1:
                raise Exception("More than 1 key for response "
                                "data dictionary is not allowed")

            # Add new classes as they appear in update
            classlstnew = response[keyval[0]]
            newclasses = list(npy.unique(classlstnew))

            # Define subset of known classes for updates
            classesk = copy.copy(classes)
            indxk = []
            for m, namek in enumerate(classesk):
                indxk.append([i for i in range(0, length)
                              if response[keyval[0]][i] == namek])

            # Handle case when new class(es) are encountered for first time
            classesuk = []
            for name in newclasses:
                if name not in classesk:  # Update classes
                    classesuk.append(name)  # Unknown classes
                    classes.append(name)  # Local scope update
                    params['classes'] = classes  # Output update
                    numclass += 1  # Local scope update
                    params['numclass'] += 1  # Output update
                    params['class_ct'].append(0)
                    indxk.append([i for i in range(0, length)
                                  if response[keyval[0]][i] == name])
                    freqs.append({k: [freqs[0][k][q] * 0
                                      for q in range(0,
                                                     len(freqs[0][k]))]
                                  for k in update.keys()})

            # Enumerate each known target class
            for j, classname in enumerate(classes):
                params['class_ct'][j] = params['class_ct'][j] + \
                                        (Counter(classlstnew)[classname])

                for i in indxk[j]:  # Iterate each row for class j

                    for key in update:  # Iterate #attributes per row

                        # Implicitly handle missing values
                        if isinstance(update[key][i], int):

                            if (update[key][i] >
                                    bin_inc[key][-1] + int_bins[key][-1]) or \
                                    (update[key][i] < int_bins[key][0]):
                                if update[key][i] > bin_inc[key][-1] \
                                        + int_bins[key][-1]:  # Add left bin
                                    newinc = (update[key][i] - int_max[key])
                                    newbin = update[key][i]
                                    bin_inc[key].append(newinc)
                                    params['bin_inc'][key] = bin_inc[key]
                                    int_bins[key].append(newbin)
                                    params['int_bins'][key] = int_bins[key]
                                    int_range[key] = abs(int_min[key] -
                                                         update[key][i])
                                    params['int_range'][key] = int_range[key]
                                    int_max[key] = update[key][i]
                                    params['int_max'][key] = int_max[key]
                                    for p, nam in enumerate(classes):
                                        freqs[p][key].append(0)
                                    freqs[j][key][-1] = 1
                                else:  # Add bin to right as needed
                                    newinc = (int_min[key] - update[key][i])
                                    newbin = update[key][i]
                                    bin_inc[key].insert(0, newinc)
                                    params['bin_inc'][key] = bin_inc[key]
                                    int_bins[key].insert(0, newbin)
                                    params['int_bins'][key] = int_bins[key]
                                    int_range[key] = abs(int_max[key] -
                                                         update[key][i])
                                    params['int_range'][key] = int_range[key]
                                    int_min[key] = update[key][i]
                                    params['int_max'][key] = int_min[key]
                                    for p, nam in enumerate(classes):
                                        freqs[p][key].insert(0, 0)
                                    freqs[j][key][0] = 1
                            else:
                                ct = 0
                                for k in range(0, len(bin_inc[key])):
                                    val = int_bins[key][k]
                                    if val <= update[key][i] <= \
                                            (val + bin_inc[key][k]):
                                        freqs[j][key][ct] += 1
                                    ct += 1
                                int_range[key] = \
                                    max([abs(params['int_max'][key] -
                                             update[key][i]),
                                         abs(update[key][i] -
                                             params['int_min'][key]),
                                         int_range[key]])
                                params['int_min'][key] = \
                                    min([update[key][i],
                                         params['int_min'][key]])
                                int_min[key] = params['int_min'][key]
                                params['int_max'][key] = \
                                    max([update[key][i],
                                         params['int_max'][key]])
                                int_max[key] = params['int_max'][key]
                                params['int_range'][key] = int_range[key]

                        # Handle NaN for float features
                        elif isinstance(update[key][i], float) \
                                and not npy.isnan(update[key][i]):

                            if (update[key][i] >
                                    (float_max[key] + self.small)) \
                                    or (update[key][i] <
                                        (float_min[key] - self.small)):
                                # Add a bin to appropriate side
                                if update[key][i] > float_max[key]:
                                    newinc = (update[key][i] -
                                              float_max[key])
                                    newbin = update[key][i]
                                    float_inc[key].append(newinc)
                                    params['float_inc'][key] = float_inc[key]
                                    float_bins[key].append(newbin)
                                    params['float_bins'][key] = float_bins[key]
                                    float_range[key] = abs(float_min[key] -
                                                           update[key][i])
                                    params['float_range'][key] = float_range[key]
                                    float_max[key] = update[key][i]
                                    params['float_max'][key] = update[key][i]
                                    for p, nam in enumerate(classes):
                                        freqs[p][key].append(0)
                                    freqs[j][key][-1] = 1
                                else:  # Add bin to right
                                    newinc = (float_min[key] - update[key][i])
                                    newbin = update[key][i]
                                    float_inc[key].insert(0, newinc)
                                    params['float_inc'][key] = float_inc[key]
                                    float_bins[key].insert(0, newbin)
                                    params['float_bins'][key] = float_bins[key]
                                    float_range[key] = abs(float_max[key] -
                                                           update[key][i])
                                    params['float_range'][key] = float_range[key]
                                    float_min[key] = update[key][i]
                                    params['float_min'][key] = update[key][i]
                                    for p, nam in enumerate(classes):
                                        freqs[p][key].insert(0, 0)
                                    freqs[j][key][0] = 1
                            else:
                                for k in range(0, len(float_inc[key])):
                                    val = float_bins[key][k]
                                    if val <= update[key][i] <= val + float_inc[key][k]:
                                        freqs[j][key][k] += 1
                                        float_range[key] = max([abs(params['float_max'][key] -
                                                                    update[key][i]),
                                                                abs(update[key][i] -
                                                                    params['float_min'][key]),
                                                                float_range[key]])
                                        float_min[key] = min([update[key][i],
                                                              params['float_min'][key]])
                                        params['float_min'][key] = float_min[key]
                                        float_max[key] = max([update[key][i],
                                                              params['float_max'][key]])
                                        params['float_max'][key] = \
                                            float_max[key]
                                        params['float_range'][key] = \
                                            float_range[key]

                        if isinstance("str", self.listmap[key]):

                            if update[key][i] is None:
                                pass
                            else:
                                if update[key][i] not in strings[key]:
                                    params['strings'][key].append(
                                        update[key][i])
                                    strings[key] = params['strings'][key]
                                    for p, nam in enumerate(classes):
                                        freqs[p][key].append(0)
                                    freqs[j][key][-1] = 1
                                else:
                                    for k, name in enumerate(strings[key]):
                                        if update[key][i] == name:
                                            freqs[j][key][k] += 1

        except Exception as e:
            raise Exception(e)

        return freqs, params

    def class_predict(self, freqs, params, test):

        """
        Naive Bayes predicter method
        Build posteriors from attributes of [test]
        set and priors from training

        Args:
        :arg freqs: (list of dicts) priors/likelihoods from prev streams
        :arg params: (dict) of parameter states from prev streams
        :arg test: (dict) current feature space stream
                    as <feature_name>:<feature_value_list>

        Returns:
        :return class_dict : dictionary containing predictions and
                             probabilities for each class prediction
        :return class_dict['class_predictions'] : predicted class for
                                                each instance in test
        :return class_dict['prob_for_predictions'] : probability for
                                each predicted class for each instance
        :return class_dict['exp_for_predictions'] : exp(probability)
                            for each predicted class for each instance
        """

        # int_range = params['int_range']
        bin_inc = params['bin_inc']
        int_bins = params['int_bins']
        # int_min = params['int_min']
        # float_range = params['float_range']
        float_inc = params['float_inc']
        float_bins = params['float_bins']
        # float_min = params['float_min']
        # numatt = params['numatt']
        numclass = params['numclass']
        classes = params['classes']
        num = params['class_ct']
        numtot = npy.sum(num)
        strings = params['strings']

        # Check feature label integrity between train and predict
        if len(test.keys()) == len(freqs[0].keys()):
            for key1 in test:
                if key1 not in freqs[0].keys():
                    raise Exception("Feature labels mismatch between "
                                    "train and predict classification data")
        else:
            raise Exception("Wrong length feature set" +
                            str(len(test.keys())) + " : " +
                            str(len(freqs[0].keys())))

        try:
            length = len(test[test.keys()[0]])
            ksm = 1
            m = 2
            classtmp = None
            probtmp = None
            exptmp = None
            class_dict = {'classes': classes,
                          'class_predictions': [],
                          'prob_for_predictions': [],
                          'exp_for_predictions': []}
            for i in range(0, length):  # loop over each row in [test]
                prob = -1000000  # Initial likelihood of each class
                for j, classname in enumerate(classes):  # Iterate class hypotheses
                    prior = (num[j] + ksm) / float((numtot + (ksm * numclass)))
                    log_p = npy.log(prior)
                    for key in test:  # Loop over number of attributes
                        if self.listmap[key] is int:  # If int, enumerate
                            ct = 0
                            for k in range(0, len(bin_inc[key])):  # Loop over range
                                val = int_bins[key][k]
                                if val <= test[key][i] <= (val + bin_inc[key][k]):
                                    inc = (freqs[j][key][ct] + (m * prior)) / float((num[j] + m))
                                    log_p += npy.log(inc)
                                ct += 1

                        elif self.listmap[key] is float:
                            for k in range(0, len(float_inc[key])):  # Loop over range
                                val = float_bins[key][k]
                                if val <= test[key][i] <= val + float_inc[key][k]:
                                    inc = (freqs[j][key][k] + (m * prior)) / float((num[j] + m))
                                    log_p += npy.log(inc)

                        elif isinstance("str", self.listmap[key]):
                            for k, name in enumerate(strings[key]):
                                if test[key][i] == name:
                                    inc = (freqs[j][key][k] + (m * prior)) / float((num[j] + m))
                                    log_p += npy.log(inc)

                    if log_p >= prob:
                        prob = copy.copy(log_p)
                        classtmp = copy.copy(classname)
                        probtmp = copy.copy(prob)
                        try:
                            exptmp = npy.exp(prob)
                        except ArithmeticError:
                            exptmp = None

                class_dict['class_predictions'].append(str(classtmp))
                class_dict['prob_for_predictions'].append(round(probtmp, 2))
                class_dict['exp_for_predictions'].append(round(exptmp, 3))

        except Exception as e:
            raise Exception(e)

        return class_dict

    @staticmethod
    def class_acc(class_dict,
                  responsedata,
                  tagname="all",
                  metadata=None):

        """
        Determine precision, recall, F-1 score and
        composite accuracy for each class predicted

        Args:
        :param class_dict: (dict) of predicted classes for
                          each instance as returned by classPredict()
        :param responsedata: (dict) of current target classes for
                          test data set
        :param tagname: (str) optional label for various class subsets
        :param metadata: (dict) optional information

        Returns:
        :return acc_dict : (dict) containing prediction error stats
                acc_dict['precision']: (float) Precision for each class
                acc_dict['recall']: (float) Recall for each class
                acc_dict['false_pos_rate']: (float) False positive rate
                acc_dict['f1_scores']: (float) F1 scores for each class
                acc_dict['class_names']: (list) all unique class names
                acc_dict['Composite_Accuracy']: (float) a class-prior weighted
                                            sum of F1 scores
        """

        # Get target class key
        keyname = responsedata.keys()[0]

        # First pull appropriate rows for
        # certain tag if metadata exists
        if metadata is not None:
            taglist = [i for i in range(0, len(metadata["tag"]))
                       if tagname in metadata["tag"][i]]
        else:
            taglist = [i for i in range(0, len(responsedata[keyname]))]

        acc_dict = {}
        crct = [0 for i in range(0, len(class_dict['classes']))]  # Counter
        fpos = [0 for i in range(0, len(class_dict['classes']))]  # Counter
        fneg = [0 for i in range(0, len(class_dict['classes']))]  # Counter
        acc_dict['precision'] = [0 for i in range(0, len(class_dict['classes']))]
        acc_dict['recall'] = [0 for i in range(0, len(class_dict['classes']))]
        acc_dict['false_pos_rate'] = [0 for i in range(0, len(class_dict['classes']))]
        acc_dict['f1_scores'] = [0 for i in range(0, len(class_dict['classes']))]
        acc_dict['class_names'] = [0 for i in range(0, len(class_dict['classes']))]

        # Check that type for classes are consistent from train and test sets
        # if type(class_dict['classes'][0]) != type(responsedata[keyname][0]):
        if not isinstance(class_dict['classes'][0], type(responsedata[keyname][0])):
            raise Exception("Response array types mismatch "
                            "between train and predict classification data")

        try:
            for i in taglist:
                for j in range(0, len(class_dict['classes'])):
                    if class_dict['class_predictions'][i] == \
                        class_dict['classes'][j] and \
                            responsedata[keyname][i] == class_dict['classes'][j]:
                        crct[j] += 1  # True positives for class j
                    elif class_dict['class_predictions'][i] == class_dict['classes'][j]:
                        fpos[j] += 1  # False positive for class j
                    else:
                        fneg[j] += 1  # False negative for class j

            # Compute precision, recall,
            # and F-1 score for each class
            class_cnt = []
            for m, classname in enumerate(class_dict['classes']):
                tmpclasslist = [responsedata[keyname][i] for i in taglist]
                class_cnt.append(Counter(tmpclasslist)[class_dict['classes'][m]])
                # Handle case for crct == 0
                if crct[m] == 0:
                    acc_dict['recall'][m] = 0.0
                    acc_dict['f1_scores'][m] = 0.0
                    acc_dict['precision'][m] = 0.0
                    acc_dict['false_pos_rate'][m] = 1.0
                    acc_dict['class_names'][m] = classname
                else:
                    precision = crct[m] / float(fpos[m] + crct[m])
                    recall_cnt = crct[m]  # Number of correct predictions
                    fp_cnt = class_cnt[m] - crct[m]  # Number of incorrect
                    acc_dict['recall'][m] += \
                        round(recall_cnt / float(class_cnt[m]), 4)  # Recall
                    acc_dict['false_pos_rate'][m] += \
                        round(fp_cnt / float(class_cnt[m]), 4)  # FP %
                    # Compute F-1 Score for each class
                    acc_dict['f1_scores'][m] = \
                        round(2 * ((precision * acc_dict['recall'][m]) /
                                   (precision + acc_dict['recall'][m])), 4)

                    acc_dict['precision'][m] = round(precision, 4)
                    acc_dict['class_names'][m] = classname

            # Compute final composite accuracy for
            # all classes from mean of F1 scores across all classes
            rlt = range(0, len(class_dict['classes']))
            acc_dict['Composite_Accuracy'] = round(
                sum([acc_dict['f1_scores'][m] * class_cnt[m]
                     for m in rlt]) / sum(class_cnt), 4)

        except Exception as e:
            raise Exception(e)

        return acc_dict


if __name__ == "__main__":
    """
    For class-level examples see /examples/bayes_tests.py
    """
