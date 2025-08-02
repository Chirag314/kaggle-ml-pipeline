import unittest
import pandas as pd
from src.main import run_pipeline

class TestPipelineIntegration(unittest.TestCase):
    def test_pipeline_runs(self):
        try:
            run_pipeline()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Pipeline failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()
