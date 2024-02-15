# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for utility functions in missing_values_processing.py."""
import pathlib
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
from auto_synthetic_data_platform import experiment_logging
from auto_synthetic_data_platform import missing_variables_processing
import numpy as np
import pandas as pd


class MissingValuesProcessingTests(parameterized.TestCase):

  def test_check_for_imbalance(self):
    input_dataframe = pd.DataFrame.from_dict({
        "categorical_column": [0, 1, 1, np.nan, 1, 1, 1, 1, 1, 1],
    })
    actual_output = missing_variables_processing.list_columns_with_missing_values(
        dataframe=input_dataframe,
    )
    self.assertTrue(actual_output, ["categorical_column"])

  def test_process_missing_values_no_missing_values(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      logger = experiment_logging.setup_logger(
          experiment_directory=pathlib.Path(temporary_directory),
          logger_name="test",
      )
      input_dataframe = pd.DataFrame.from_dict({
          "categorical_column": [0, 1],
          "numerical_column": [0.1, 0.2],
      })
      actual_output = missing_variables_processing.process_missing_values(
          dataframe=input_dataframe,
          missing_values_preprocessing_method="drop",
          logger=logger,
      )
      dataframe_equal = actual_output.equals(input_dataframe)
      expected_message = "No columns with missing values detected"
      log_path = pathlib.Path(temporary_directory).joinpath(
          "logs/test_logs.log"
      )
      with log_path.open() as logs:
        logged_message = logs.readlines()
      messages_equal = expected_message in logged_message[0]
      self.assertTrue(all([dataframe_equal, messages_equal]))

  @parameterized.named_parameters(
      dict(
          testcase_name="drop_method",
          missing_values_preprocessing_method="drop",
          input_dataframe=pd.DataFrame.from_dict({
              "categorical_column": [0, np.nan],
              "numerical_column": [0.1, 0.2],
          }),
          output_dataframe=pd.DataFrame.from_dict({
              "categorical_column": [0.0],
              "numerical_column": [0.1],
          }),
          expected_second_log_excerpt=(
              "Preprocesssing the missing values using the specified method:"
              " 'drop'."
          ),
      ),
      dict(
          testcase_name="unrecommended_cardinality",
          missing_values_preprocessing_method="mode",
          input_dataframe=pd.DataFrame.from_dict({
              "categorical_column": [0, np.nan],
              "numerical_column": [0.1, 0.2],
          }),
          output_dataframe=pd.DataFrame.from_dict({
              "categorical_column": [0.0, 0.0],
              "numerical_column": [0.1, 0.2],
          }),
          expected_second_log_excerpt=(
              "Preprocesssing the missing values using the specified method:"
              " 'mode'."
          ),
      ),
  )
  def test_process_missing_values_no_missing_values(
      self,
      missing_values_preprocessing_method,
      input_dataframe,
      output_dataframe,
      expected_second_log_excerpt,
  ):
    with tempfile.TemporaryDirectory() as temporary_directory:
      logger = experiment_logging.setup_logger(
          experiment_directory=pathlib.Path(temporary_directory),
          logger_name="test",
      )
      actual_output = missing_variables_processing.process_missing_values(
          dataframe=input_dataframe,
          missing_values_preprocessing_method=missing_values_preprocessing_method,
          logger=logger,
      )
      dataframe_equal = actual_output.equals(output_dataframe)
      log_path = pathlib.Path(temporary_directory).joinpath(
          "logs/test_logs.log"
      )
      with log_path.open() as logs:
        logged_message = logs.readlines()
      first_message_equal = (
          "The input dataframe has missing values in columns:"
          " categorical_column."
          in logged_message[0]
      )
      second_message_equal = expected_second_log_excerpt in logged_message[1]
      self.assertTrue(
          all([dataframe_equal, first_message_equal, second_message_equal])
      )

  def test_process_missing_values_value_error(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      logger = experiment_logging.setup_logger(
          experiment_directory=pathlib.Path(temporary_directory),
          logger_name="test",
      )
      input_dataframe = pd.DataFrame.from_dict({
          "categorical_column": [0, np.nan],
          "numerical_column": [0.1, 0.2],
      })
      with self.assertRaisesRegex(ValueError, "Unknown missing values method"):
        missing_variables_processing.process_missing_values(
            dataframe=input_dataframe,
            missing_values_preprocessing_method="unknown_method",
            logger=logger,
        )


if __name__ == "__main__":
  absltest.main()
