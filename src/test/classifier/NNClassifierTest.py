import logging
import os
import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from classifier.NNClassifier import NNClassifier


class NNClassifierTest(unittest.TestCase):
    TEST_DATA_DIR = None

    @classmethod
    def setUpClass(cls) -> None:
        logging.basicConfig(level=logging.INFO)
        if not cls.TEST_DATA_DIR:
            cls.TEST_DATA_DIR = f"../../../data/testing/{__class__.__name__}"
        os.makedirs(cls.TEST_DATA_DIR, exist_ok=True)

    def test_training(self):

        confidence = 0.7    # Confidence level for prediction

        # Create a simple sample dataset
        np.random.seed(42)  # For reproducibility

        num_samples = 1000
        num_features = 20
        num_classes = 5

        # Generate random data
        x = np.random.randn(num_samples, num_features)
        # Generate random labels from 0 to 4 (5 classes)
        y = np.random.randint(0, num_classes, num_samples)

        # Alternate y so the result can easily be eyeballed. (If 1st feature > 1, it is in class 1.)
        mask = x[:, 0] > 1
        y[mask] = 1
        y[~mask] = np.where(y[~mask] == 1, 0, y[~mask])

        # Split into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Train the model
        classifier = NNClassifier(num_features=num_features, num_classes=num_classes, confidence=confidence)
        classifier.train_with(x_train, y_train, x_test, y_test)

        # Create data for unit tests.
        # Data that fall into class 1
        np.random.seed(23)  # For reproducibility
        class_one = np.random.randn(10, num_features)
        class_one[:, 0] = np.abs(class_one[:, 0]) + 1
        # Data for any classes
        class_any = np.random.randn(20, num_features)
        # Put together a test data
        test_data = np.concatenate((class_one, class_any), axis=0)
        # Infer from test data
        test_infer = classifier.infer(test_data)

        # Put results and test data together for comparison
        compare = np.c_[test_infer, test_data[:, 0] > 1, test_data[:, 0]]
        print(compare)

        # Class 1 should have high confidence, all others should be low because of random features
        # All predicted class 1 (0th column of 'compare' == 1) should have 1st feature > 1 (2nd column of 'compare')
        correct_class_one_feature = np.all(compare[(compare[:, 0] == 1), 2] == 1)
        self.assertTrue(correct_class_one_feature)

        # If all unclassified should have lower probs than the confidence level
        unclassified_low_confidence = np.all(compare[(compare[:, 0] == -1), 1] < confidence)
        self.assertTrue(unclassified_low_confidence)

        # Save to a file
        model_file = os.path.join(self.TEST_DATA_DIR, "basic_test")
        classifier.save(model_file)

        # Load it back and proof it works the same
        classifier2 = NNClassifier.load(model_file)
        test_infer2 = classifier2.infer(test_data)
        self.assertTrue(np.allclose(test_infer, test_infer2))


if __name__ == '__main__':
    unittest.main()
