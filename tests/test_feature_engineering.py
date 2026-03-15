import unittest
import pandas as pd
import numpy as np
from kaggle_ml_pipeline.features.tabular import FeatureGenerator

class TestFeatureGenerator(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "age": [25, 30, np.nan, 40],
            "income": [1000, 1500, 2000, 2500],
            "job": ["engineer", "doctor", "artist", "lawyer"]
        })
        self.y = pd.Series([1, 0, 1, 0])
        self.numerical = ["age", "income"]
        self.categorical = ["job"]
        self.fg = FeatureGenerator(self.numerical, self.categorical)
        self.fg.fit(self.data, self.y)

    def test_transform(self):
        result = self.fg.transform(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.notna().all().all())

if __name__ == "__main__":
    unittest.main()
