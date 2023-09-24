"""Tests for utility functions in categorical_variables_processing.py."""
import pathlib
import tempfile
from absl.testing import parameterized
from auto_synthetic_data_platform import categorical_variables_processing
from auto_synthetic_data_platform import experiment_logging
import numpy as np
import pandas as pd


class CategoricalVariablesProcessingTests(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="mild_imbalance",
          test_dataframe=pd.DataFrame.from_dict({
              "categorical_column": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
          }),
          expected_log_excerpt="A mild class imbalance detected",
      ),
      dict(
          testcase_name="moderate_imbalance",
          test_dataframe=pd.DataFrame.from_dict({
              "categorical_column": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          }),
          expected_log_excerpt="A moderate class imbalance detected",
      ),
      dict(
          testcase_name="severe_imbalance",
          test_dataframe=pd.DataFrame.from_dict({
              "categorical_column": [0] + [1] * 100,
          }),
          expected_log_excerpt="An extreme class imbalance detected",
      ),
  )
  def test_check_for_imbalance(self, test_dataframe, expected_log_excerpt):
    with tempfile.TemporaryDirectory() as temporary_directory:
      logger = experiment_logging.setup_logger(
          experiment_directory=pathlib.Path(temporary_directory),
          logger_name="test",
      )
      categorical_variables_processing.check_for_imbalance(
          categorical_column_name="categorical_column",
          categorical_column_data=test_dataframe["categorical_column"],
          logger=logger,
      )
      log_path = pathlib.Path(temporary_directory).joinpath(
          "logs/test_logs.log"
      )
      with log_path.open() as logs:
        logged_message = logs.readlines()
      self.assertTrue(expected_log_excerpt in logged_message[0])

  @parameterized.named_parameters(
      dict(
          testcase_name="exceeds_recommended_maximum_dimensionality",
          test_dataframe=pd.DataFrame.from_dict({
              "categorical_column": np.arange(0, 1001),
          }),
          expected_log_excerpt="has a higher dimensionality than",
      ),
      dict(
          testcase_name="unrecommended_cardinality",
          test_dataframe=pd.DataFrame.from_dict({
              "categorical_column": np.arange(0, 49),
          }),
          expected_log_excerpt="recommended column cardinality range is 50-100",
      ),
  )
  def test_check_categorical_column(self, test_dataframe, expected_log_excerpt):
    with tempfile.TemporaryDirectory() as temporary_directory:
      logger = experiment_logging.setup_logger(
          experiment_directory=pathlib.Path(temporary_directory),
          logger_name="test",
      )
      categorical_variables_processing.check_categorical_column(
          categorical_column_name="categorical_column",
          categorical_column_data=test_dataframe["categorical_column"],
          logger=logger,
      )
      log_path = pathlib.Path(temporary_directory).joinpath(
          "logs/test_logs.log"
      )
      with log_path.open() as logs:
        logged_message = logs.readlines()
      self.assertTrue(expected_log_excerpt in logged_message[0])
