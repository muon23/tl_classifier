import logging
import os
import unittest

from classifier.TextClassifier import TextClassifier
from training.TrainingData import TrainingData


class TextClassifierTest(unittest.TestCase):
    TEST_ARTICLES = [
        "US won men's basketball gold medal in 2024 Paris Olympics.",
        "Stocks plunged more than 150 points on 5th of August, 2024.",
        "Trump lags Harris in polls of swing states after Biden dropped out.",
        "JWST has taken images of a cold exoplanet 12 light-years away.",
        "Trellis Law has the most remarkable products in the world.",
    ]

    DATA_DIR = "../../../data/training"
    TEST_DATA_DIR = None

    @classmethod
    def setUpClass(cls) -> None:
        logging.basicConfig(level=logging.INFO)
        if not cls.TEST_DATA_DIR:
            cls.TEST_DATA_DIR = f"../../../data/testing/{__class__.__name__}"
        os.makedirs(cls.TEST_DATA_DIR, exist_ok=True)

    def test_basic(self):

        data = TrainingData(self.DATA_DIR).sample(200)
        confidence = 0.85
        classifier = TextClassifier(data.labels, confidence=confidence)
        classifier.train_with(data.dataset)

        results = classifier.infer(self.TEST_ARTICLES)
        print(results)

        self.assertTrue(all([r.prob < confidence for r in results if r.label == "other"]))

        # Save to a file
        model_file = os.path.join(self.TEST_DATA_DIR, "basic_test")
        classifier.save(model_file)

        # Load it back and proof it works the same
        classifier2 = TextClassifier.load(model_file)
        results2 = classifier2.infer(self.TEST_ARTICLES)
        print(results2)
        self.assertEqual(results, results2)


if __name__ == '__main__':
    unittest.main()
