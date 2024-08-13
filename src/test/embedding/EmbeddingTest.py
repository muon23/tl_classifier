import unittest
import embedding


class EmbeddingTest(unittest.TestCase):

    def test_bert(self):
        embedding_model = embedding.of("bert")

        texts = [
            "US won men's basketball gold medal in 2024 Paris Olympics.",
            "Stock plunged more than 150 points on 5th of August, 2024.",
            "Trump lags Harris in polls of swing states.",
        ]
        result = embedding_model.embed(texts)

        print(result.shape)
        self.assertEqual(result.shape, (len(texts), embedding_model.dimensions()))


if __name__ == '__main__':
    unittest.main()
