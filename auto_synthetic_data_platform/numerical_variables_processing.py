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

"""A numerical values processing module of the Google EMEA gPS Data Science Auto Synthetic Data Platform."""
import logging
from typing import Any, Mapping
import numpy as np
import pandas as pd


def remove_column_outliers(
    *,
    dataframe: pd.DataFrame,
    column_name: str,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> pd.DataFrame:
  """Returns a dataframe with removed rows containing the column's outliers.

  Args:
    dataframe: A dataframe to check for outliers.
    column_name: A numerical column name to check for outliers.
    lower_percentile: A lower precentile boundary. All the values below the
      lower limit marked by this percentile will be removed.
    upper_percentile: An upper precentile boundary. All the values over the
      upper limit marked by this percentile will be removed.
  """
  lower_limit, upper_limit = np.percentile(
      a=dataframe[column_name], q=[lower_percentile, upper_percentile]
  )
  return dataframe[
      (dataframe[column_name] > lower_limit)
      & (dataframe[column_name] < upper_limit)
  ]


def process_numerical_columns(
    *,
    dataframe: pd.DataFrame,
    column_metadata: Mapping[str, list[Any]],
    remove_numerical_outliers: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
  """Preprocesses numerical columns of the dataframe.

  Args:
    dataframe: A dataframe for the preprocessing.
    column_metadata: A mapping between "categorical" and "numerical" data types
      and column names.
    remove_numerical_outliers: An indicator if to remove numerical outliers from
      the numeric columns.
    logger: A logger object to log and save preprocessing messages.

  Returns:
    A dataframe with processed numerical columns.
  """
  numerical_column_names = column_metadata.get("numerical", None)
  if not numerical_column_names:
    logger.info("No numerical columns are specified for the dataframe.")
    return dataframe
  if remove_numerical_outliers:
    for column_name in numerical_column_names:
      dataframe = remove_column_outliers(
          dataframe=dataframe, column_name=column_name
      )
  return dataframe
