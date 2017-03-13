#!/usr/bin/python

# Externals
import unittest
from code import Classifier


class TestClassifier(unittest.TestCase):
    def test_validate_features(self):

        # Test for data type validation
        features = {'1': [1.0, 1], '2': ['hello', 1.0], '3': [1, 'hello']}
        learner = Classifier()
        learner.validate_features(features)
        for key in learner.listmap:
            if isinstance(learner.listmap[key], float):
                self.assertEquals(key, '1')
            elif isinstance(learner.listmap[key], int):
                self.assertEquals(key, '3')
            elif isinstance(learner.listmap[key], (str, unicode)):
                self.assertEquals(key, '2')

        # Test for raised exception on None type input
        with self.assertRaises(Exception) as context:
            learner.validate_features({'null_value': [None]})
        self.assertTrue("Input features must be type int, float, or string"
                        in str(context.exception))

    def test_class_learn(self):
        features = {'1': [1.0, 2.0], '2': ['hello', 'Goodbye'],
                    '3': [1, 3]}
        response = {'response': ['Yes', 'No']}
        expected_output = {
            'numclass': 2, 'float_inc': {'1': [0.16666666666666674,
                                               0.16666666666666652,
                                               0.16666666666666674,
                                               0.16666666666666652,
                                               0.16666666666666674,
                                               0.16666666666666674]},
            'int_max': {'3': 3}, 'int_range': {'3': 2}, 'class_ct': [1, 1],
            'float_min': {'1': 1.0}, 'int_bins': {'3': [1, 3, 5, 7, 9, 11, 13]},
            'float_bins': {'1': [1.0,
                                 1.1666666666666667,
                                 1.3333333333333333,
                                 1.5,
                                 1.6666666666666665,
                                 1.8333333333333333,
                                 2.0]}, 'numatt': 3, 'bin_inc': {'3': [2, 2, 2, 2, 2, 2]},
            'int_min': {'3': 1}, 'classes': ['Yes', 'No'], 'float_range': {'1': 1.0},
            'float_max': {'1': 2.0}, 'strings': {'2': ['hello', 'Goodbye']}
        }

        learner = Classifier()
        freq, param = learner.class_learn(features, response)
        for key in param:
            self.assertEqual(param[key], expected_output[key])

    def test_class_update(self):
        pass

    def test_class_predict(self):
        pass

    def test_class_acc(self):
        pass

if __name__ == '__main__':

    unittest.main()
