import logging
import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from classifier.NNClassifier import NNClassifier


class NNClassifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.basicConfig(level=logging.INFO)

    def test_training(self):

        # Create a simple sample dataset
        np.random.seed(42)  # For reproducibility

        num_samples = 1000
        num_features = 20
        num_classes = 5

        # Generate random data
        X = np.random.randn(num_samples, num_features)
        # Generate random labels from 0 to 4 (5 classes)
        y = np.random.randint(0, num_classes, num_samples)

        # Alternate y so the result can easily be verified.
        mask = X[:, 0] > 1
        y[mask] = 1
        y[~mask] = np.where(y[~mask] == 1, 0, y[~mask])

        # Split into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifier = NNClassifier(num_featuers=num_features, num_classes=num_classes)
        classifier.train_with(x_train, y_train, x_test, y_test)

        # Make a bunch of test data all in class 1
        class_one = np.random.randn(10, num_features)
        class_one[:, 0] = np.abs(class_one[:, 0]) + 1

        # Make any class
        class_any = np.random.randn(20, num_features)

        test_data = np.concatenate((class_one, class_any), axis=0)
        test_infer = classifier.infer(test_data)

        print(np.c_[test_infer, test_data[:, 0] > 1, test_data[:, 0]])


if __name__ == '__main__':
    unittest.main()
