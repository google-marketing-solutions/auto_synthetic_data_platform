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

"""Tests for utility functions in experiment_logging.py."""
import pathlib
import tempfile
from absl.testing import absltest
from auto_synthetic_data_platform import experiment_logging
from auto_synthetic_data_platform import numerical_variables_processing
import pandas as pd

_DATAFRAME = pd.DataFrame.from_dict({
    "categorical_column": [1, 2, 3, 4],
    "numerical_column": [1.01, 2.91, 3.05, 1000000],
})
_EXPECTED_DATAFRAME = pd.DataFrame.from_dict(
    {"categorical_column": [2, 3], "numerical_column": [2.91, 3.05]}
)


class NumericalValuesProcessingTests(absltest.TestCase):

  def test_remove_column_outliers(self):
    actual_output = numerical_variables_processing.remove_column_outliers(
        dataframe=_DATAFRAME, column_name="numerical_column"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_output, _EXPECTED_DATAFRAME)

  def test_process_numerical_columns_no_numerical_columns(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      column_metadata = {"categorical": ["categorical_column"]}
      logger = experiment_logging.setup_logger(
          experiment_directory=pathlib.Path(temporary_directory),
          logger_name="test",
      )
      actual_output = numerical_variables_processing.process_numerical_columns(
          dataframe=_DATAFRAME,
          column_metadata=column_metadata,
          remove_numerical_outliers=True,
          logger=logger,
      )
      dataframe_equal = actual_output.equals(_DATAFRAME)
      expected_message = "No numerical columns are specified for the dataframe."
      log_path = pathlib.Path(temporary_directory).joinpath(
          "logs/test_logs.log"
      )
      with log_path.open() as logs:
        logged_message = logs.readlines()
      messages_equal = expected_message in logged_message[0]
      self.assertTrue(all([dataframe_equal, messages_equal]))

  def test_process_numerical_columns_with_numerical_columns(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      column_metadata = {"numerical": ["numerical_column"]}
      logger = experiment_logging.setup_logger(
          experiment_directory=pathlib.Path(temporary_directory),
          logger_name="test",
      )
      actual_output = numerical_variables_processing.process_numerical_columns(
          dataframe=_DATAFRAME,
          column_metadata=column_metadata,
          remove_numerical_outliers=True,
          logger=logger,
      ).reset_index(drop=True)
      pd.testing.assert_frame_equal(actual_output, _EXPECTED_DATAFRAME)


if __name__ == "__main__":
  absltest.main()
