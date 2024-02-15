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

"""An end-to-end real data preprocessing module of the Google EMEA gPS Data Science Auto Synthetic Data Platform."""
import functools
import logging
import pathlib
from typing import Any, Final, Mapping
from auto_synthetic_data_platform import categorical_variables_processing
from auto_synthetic_data_platform import experiment_logging
from auto_synthetic_data_platform import missing_variables_processing
from auto_synthetic_data_platform import numerical_variables_processing
import pandas as pd

_RECOMMENDED_MAXIMUM_FEATURE_NUMBER: Final[int] = 100
_DEFAULT_PREPROCESS_METADATA: Final[dict] = dict(
    remove_duplicates=True,
    preprocess_missing_values=True,
    missing_values_preprocessing_method="drop",
    remove_numerical_outliers=True,
)


def preprocess_dataframe(
    *,
    dataframe: pd.DataFrame,
    column_metadata: Mapping[str, list[Any]],
    logger: logging.Logger,
    preprocess_metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
  """Returns a preprocessed dataframe for synthetic data model training.

  Args:
    dataframe: A dataframe to preprocess.
    column_metadata: A mapping between "categorical" and "numerical" data types
      and column names.
    preprocess_metadata: A sequence with mappings between preprocessing variable
      names and their values used to preprocess the input dataframe with the
      real data before training any synthetic data models.
    logger: A logger object to log and save preprocessing messages.
  """
  if not preprocess_metadata:
    preprocess_metadata = _DEFAULT_PREPROCESS_METADATA
  if preprocess_metadata.get("remove_duplicates", True):
    dataframe.drop_duplicates(inplace=True)
  if preprocess_metadata.get("preprocess_missing_values", True):
    dataframe = missing_variables_processing.process_missing_values(
        dataframe=dataframe,
        missing_values_preprocessing_method=preprocess_metadata.get(
            "missing_values_method", "drop"
        ),
        logger=logger,
    )
  dataframe_with_processed_numerical_values = (
      numerical_variables_processing.process_numerical_columns(
          dataframe=dataframe,
          column_metadata=column_metadata,
          remove_numerical_outliers=preprocess_metadata.get(
              "remove_numerical_outliers", True
          ),
          logger=logger,
      )
  )
  processed_dataframe = (
      categorical_variables_processing.process_categorical_columns(
          dataframe=dataframe_with_processed_numerical_values,
          column_metadata=column_metadata,
          logger=logger,
      )
  )
  if processed_dataframe.shape[1] > _RECOMMENDED_MAXIMUM_FEATURE_NUMBER:
    logger.warning(
        "The dataframe exceeds the recommended maximum feature number."
        " It's likely to lower the synthetic data model quality."
    )
  return processed_dataframe


class Preprocessor:
  """Analyses and preprocess the input dataframe.

  Attributes:
    dataframe_path: A path to the dataframe with the real data. The package will
      use it to train a synthetic data model and then to generate a fake version
      of this dataframe.
    experiment_directory: A path to the experiment directory where all the
      experiment artifacts will be saved.
    column_metadata: A mapping between "categorical" and "numerical" data types
      and column names.
    preprocess_metadata: A sequence with mappings between preprocessing variable
      names and their values used to preprocess the input dataframe with the
      real data before training any synthetic data models.
    input_dataframe: An input dataframe with the real data.
    output_dataframe: An output dataframe with the preprocessed real data.
    logger: A logger object to log and save preprocessing messages.
  """

  def __init__(
      self,
      *,
      dataframe: pathlib.Path | pd.DataFrame,
      experiment_directory: pathlib.Path,
      column_metadata: Mapping[str, list[Any]],
      preprocess_metadata: Mapping[str, Any] | None = None,
  ) -> None:
    """Initializes the Preprocessor class.

    Args:
      dataframe: A path to the dataframe with the real data or a datframe with real data. The package
        will use it to train a synthetic data model and then to generate a fake
        version of this dataframe.
      experiment_directory: A path to the experiment directory where all the
        experiment artifacts will be saved.
      column_metadata: A mapping between "categorical" and "numerical" data
        types and column names.
      preprocess_metadata: A sequence with mappings between preprocessing
        variable names and their values used to preprocess the input dataframe
        with the real data before training any synthetic data models.
    """
    self.dataframe = dataframe
    self.experiment_directory = experiment_directory
    self.column_metadata = column_metadata
    self.preprocess_metadata = preprocess_metadata

  @functools.cached_property
  def input_dataframe(self) -> pd.DataFrame:
    """Returns the input dataframe with the real data."""
    if isinstance(self.dataframe, pd.DataFrame):
      return self.dataframe
    return pd.read_csv(self.dataframe_path)

  @functools.cached_property
  def logger(self) -> logging.Logger:
    """Returns a logger object for logging and saving preprocessing messages."""
    return experiment_logging.setup_logger(
        experiment_directory=self.experiment_directory,
        logger_name="preprocessing",
    )

  @functools.cached_property
  def output_dataframe(self) -> pd.DataFrame:
    """Returns the output dataframe with the preprocessed real data."""
    return preprocess_dataframe(
        dataframe=self.input_dataframe,
        column_metadata=self.column_metadata,
        preprocess_metadata=self.preprocess_metadata,
        logger=self.logger,
    )
