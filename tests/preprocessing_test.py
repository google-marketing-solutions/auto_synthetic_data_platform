"""Tests for utility functions in preprocessing.py."""
import pathlib
import tempfile
from absl.testing import absltest
from auto_synthetic_data_platform import experiment_logging
from auto_synthetic_data_platform import preprocessing
import numpy as np
import pandas as pd

_DATAFRAME = pd.DataFrame.from_dict({
    "categorical_column": [0, 1],
    "numerical_column": [0.1, 0.2],
})


class PreprocessingTests(absltest.TestCase):

  def test_preprocess_dataframe_remove_duplicates(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      logger = experiment_logging.setup_logger(
          experiment_directory=pathlib.Path(temporary_directory),
          logger_name="test",
      )
      input_dataframe = pd.DataFrame.from_dict({
          "categorical_column": [0, 0],
          "numerical_column": [0.1, 0.1],
      })
      actual_output = preprocessing.preprocess_dataframe(
          dataframe=input_dataframe,
          column_metadata={
              "categorical": ["categorical_column"],
              "numerical": ["numerical_column"],
          },
          preprocess_metadata={"remove_numerical_outliers": False},
          logger=logger,
      )
      expected_output = pd.DataFrame.from_dict({
          "categorical_column": [0],
          "numerical_column": [0.1],
      })
      pd.testing.assert_frame_equal(actual_output, expected_output)

  def test_preprocess_dataframe_preprocess_missing_values(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      logger = experiment_logging.setup_logger(
          experiment_directory=pathlib.Path(temporary_directory),
          logger_name="test",
      )
      input_dataframe = pd.DataFrame.from_dict({
          "categorical_column": [0, np.nan],
          "numerical_column": [0.1, 0.1],
      })
      actual_output = preprocessing.preprocess_dataframe(
          dataframe=input_dataframe,
          column_metadata={
              "categorical": ["categorical_column"],
              "numerical": ["numerical_column"],
          },
          preprocess_metadata={
              "remove_numerical_outliers": False,
              "preprocess_missing_values": True,
          },
          logger=logger,
      )
      expected_output = pd.DataFrame.from_dict({
          "categorical_column": [0.0],
          "numerical_column": [0.1],
      })
      pd.testing.assert_frame_equal(actual_output, expected_output)

  def test_preprocess_dataframe_exceeds_recommended_maximum_feature_number(
      self,
  ):
    with tempfile.TemporaryDirectory() as temporary_directory:
      logger = experiment_logging.setup_logger(
          experiment_directory=pathlib.Path(temporary_directory),
          logger_name="test",
      )
      input_dataframe = pd.DataFrame([np.arange(101)])
      actual_output = preprocessing.preprocess_dataframe(
          dataframe=input_dataframe,
          column_metadata={"categorical": [*range(0, 101, 1)]},
          logger=logger,
      )
      dataframe_equal = actual_output.equals(input_dataframe)
      log_path = pathlib.Path(temporary_directory).joinpath(
          "logs/test_logs.log"
      )
      with log_path.open() as logs:
        logged_message = logs.readlines()
      messages_equal = (
          "The dataframe exceeds the recommended maximum feature number."
          in logged_message[-1]
      )
      self.assertTrue(all([dataframe_equal, messages_equal]))
