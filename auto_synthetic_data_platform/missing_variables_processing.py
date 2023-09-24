# Copyright 2023 Google LLC.
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

"""A missing values processing module of the Google EMEA gPS Data Science Auto Synthetic Data Platform."""
import logging
import numpy as np
import pandas as pd
from typing import Literal


def list_columns_with_missing_values(*, dataframe: pd.DataFrame) -> list[str]:
  """Returns an sequence of columns with missing values in the dataframe.

  Args:
    dataframe: A dataframe to check for missing values.
  """
  dataframe.replace("", np.nan, inplace=True)
  return dataframe.columns[dataframe.isnull().any()].tolist()


def process_missing_values(
    *,
    dataframe: pd.DataFrame,
    missing_values_preprocessing_method: Literal["drop", "mode"],
    logger: logging.Logger,
) -> pd.DataFrame:
  """Processes missing values from the dataframe using the specified method.

  Args:
    dataframe: A dataframe with missing values.
    missing_values_preprocessing_method: A preprocessing method describing how
      to process missing values. Possible methods are "drop" and "mode".
    logger: A logger object to log and save preprocessing messages.

  Returns:
    A dataframe with processed missing values.

  Raises:
   ValueError: An error if a user tries to use an unknown
    'missing_values_preprocessing_method'.
  """
  columns_with_missing_values = list_columns_with_missing_values(
      dataframe=dataframe
  )
  if not columns_with_missing_values:
    logger.info("No columns with missing values detected.")
    return dataframe
  logger.warning(
      "The input dataframe has missing values in columns: %s."
      % ", ".join(columns_with_missing_values)
  )
  if missing_values_preprocessing_method == "drop":
    dataframe.dropna(axis=0, how="any", inplace=True)
  elif missing_values_preprocessing_method == "mode":
    dataframe.fillna(dataframe.mode().iloc[0], inplace=True)
  else:
    raise ValueError(
        f"Unknown missing values method: {missing_values_preprocessing_method}"
    )
  logger.warning(
      "Preprocesssing the missing values using the specified method: '%s'."
      % missing_values_preprocessing_method
  )
  return dataframe
