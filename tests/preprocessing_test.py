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

"""Tests for utility functions in preprocessing.py."""
import logging
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


class PreprocessorTests(absltest.TestCase):

  def test_preprocessor_input_dataframe_from_file(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      dataframe_path = pathlib.Path(temporary_directory).joinpath("test.csv")
      _DATAFRAME.to_csv(dataframe_path, index=False)
      preprocessor = preprocessing.Preprocessor(
          dataframe_path=pathlib.Path(dataframe_path),
          experiment_directory=temporary_directory,
          column_metadata={
              "categorical": ["categorical_column"],
              "numerical": ["numerical_column"],
          },
      )
      pd.testing.assert_frame_equal(preprocessor.input_dataframe, _DATAFRAME)

  def test_preprocessor_input_dataframe(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      preprocessor = preprocessing.Preprocessor(
          dataframe_path=_DATAFRAME,
          experiment_directory=temporary_directory,
          column_metadata={
              "categorical": ["categorical_column"],
              "numerical": ["numerical_column"],
          },
      )
      pd.testing.assert_frame_equal(preprocessor.input_dataframe, _DATAFRAME)

  def test_preprocessor_logger(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      dataframe_path = pathlib.Path(temporary_directory).joinpath("test.csv")
      _DATAFRAME.to_csv(dataframe_path, index=False)
      preprocessor = preprocessing.Preprocessor(
          dataframe_path=dataframe_path,
          experiment_directory=pathlib.Path(temporary_directory),
          column_metadata={
              "categorical": ["categorical_column"],
              "numerical": ["numerical_column"],
          },
      )
      self.assertIsInstance(preprocessor.logger, logging.Logger)

  def test_preprocessor_output_dataframe(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      dataframe_path = pathlib.Path(temporary_directory).joinpath("test.csv")
      _DATAFRAME.to_csv(dataframe_path, index=False)
      preprocessor = preprocessing.Preprocessor(
          dataframe_path=dataframe_path,
          experiment_directory=pathlib.Path(temporary_directory),
          column_metadata={
              "categorical": ["categorical_column"],
              "numerical": ["numerical_column"],
          },
      )
      pd.testing.assert_frame_equal(preprocessor.input_dataframe, _DATAFRAME)


if __name__ == "__main__":
  absltest.main()
