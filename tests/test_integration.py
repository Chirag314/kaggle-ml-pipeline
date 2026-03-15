import unittest
from pathlib import Path
import tempfile

from kaggle_ml_pipeline.pipeline import run_pipeline

class TestPipelineIntegration(unittest.TestCase):
    def test_pipeline_runs(self):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_file = run_pipeline(output_path=str(Path(tmpdir) / "submission.csv"))
                self.assertTrue(output_file.exists())
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Pipeline failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()
