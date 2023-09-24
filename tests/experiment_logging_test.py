"""Tests for utility functions in experiment_logging.py."""
import logging
import pathlib
import tempfile
from absl.testing import absltest
from auto_synthetic_data_platform import experiment_logging


class ExperimentLoggingTests(absltest.TestCase):

  def test_setup_logger(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      logger = experiment_logging.setup_logger(
          experiment_directory=pathlib.Path(temporary_directory),
          logger_name="test",
      )
      self.assertIsInstance(logger, logging.Logger)


if __name__ == "__main__":
  absltest.main()
