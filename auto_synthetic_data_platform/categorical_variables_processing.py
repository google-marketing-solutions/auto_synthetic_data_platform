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

"""A categorical values processing module of the Google EMEA gPS Data Science Auto Synthetic Data Platform."""
import logging
from typing import Any, Final, Mapping
import pandas as pd

_MILD_IMBALANCE_UPPER_LIMIT: Final[float] = 0.4
_MILD_IMBALANCE_LOWER_LIMIT: Final[float] = 0.2
_MODERATE_IMBALANCE_LOWER_LIMIT: Final[float] = 0.01
_RECOMMENDED_MAXIMUM_DIMENSIONALITY: Final[int] = 1000


def check_for_imbalance(
    *,
    categorical_column_name: str,
    categorical_column_data: pd.Series,
    logger: logging.Logger,
) -> None:
  """Checks for class imbalances in a categorical column.

  Args:
    categorical_column_name: A categorical column name.
    categorical_column_data: A categorical column data.
    logger: A logger object to log and save preprocessing messages.
  """
  percentage_distributions = categorical_column_data.value_counts(
      normalize=True
  )
  mild_imbalance_occurences = percentage_distributions[
      (percentage_distributions > _MILD_IMBALANCE_LOWER_LIMIT)
      & (percentage_distributions <= _MILD_IMBALANCE_UPPER_LIMIT)
  ]
  moderate_imbalance_occurences = percentage_distributions[
      (percentage_distributions > _MODERATE_IMBALANCE_LOWER_LIMIT)
      & (percentage_distributions <= _MILD_IMBALANCE_LOWER_LIMIT)
  ]
  severe_imbalance_occurences = percentage_distributions[
      (percentage_distributions < _MODERATE_IMBALANCE_LOWER_LIMIT)
  ]
  if mild_imbalance_occurences.any():
    imbalance_class_names = [
        str(class_name)
        for class_name in mild_imbalance_occurences.reset_index(0)
        .iloc[:, 0]
        .values.tolist()
    ]
    logger.warning(
        "A mild class imbalance detected in the '%s' column for variables: %s."
        % (categorical_column_name, ", ".join(imbalance_class_names))
    )
  if moderate_imbalance_occurences.any():
    imbalance_class_names = [
        str(class_name)
        for class_name in moderate_imbalance_occurences.reset_index(0)
        .iloc[:, 0]
        .values.tolist()
    ]
    logger.warning(
        "A moderate class imbalance detected in the '%s' column for"
        " variables: %s."
        % (categorical_column_name, ", ".join(imbalance_class_names))
    )
  elif severe_imbalance_occurences.any():
    imbalance_class_names = [
        str(class_name)
        for class_name in severe_imbalance_occurences.reset_index(0)
        .iloc[:, 0]
        .values.tolist()
    ]
    logger.warning(
        "An extreme class imbalance detected in the '%s' column for"
        " variables: %s."
        % (categorical_column_name, ", ".join(imbalance_class_names))
    )


def check_categorical_column(
    *,
    categorical_column_name: str,
    categorical_column_data: pd.Series,
    logger: logging.Logger,
) -> None:
  """Runs checks on a categorical column.

  Args:
    categorical_column_name: A categorical column name.
    categorical_column_data: A categorical column data.
    logger: A logger object to log and save preprocessing messages.
  """
  column_dimensions = categorical_column_data.nunique()
  if column_dimensions > _RECOMMENDED_MAXIMUM_DIMENSIONALITY:
    logger.warning(
        "The categorical column '%s' has a higher dimensionality than"
        " recommended, %d > %r."
        % (
            categorical_column_name,
            column_dimensions,
            _RECOMMENDED_MAXIMUM_DIMENSIONALITY,
        )
    )
  if column_dimensions < 50 or column_dimensions > 100:
    logger.warning(
        "The categorical column '%s' dimensionality is %s. The recommended"
        " column cardinality range is 50-100. Real and synthetic distribution"
        " metrics might get adversely affected."
        % (categorical_column_name, column_dimensions)
    )
  check_for_imbalance(
      categorical_column_name=categorical_column_name,
      categorical_column_data=categorical_column_data,
      logger=logger,
  )


def process_categorical_columns(
    *,
    dataframe: pd.DataFrame,
    column_metadata: Mapping[str, list[Any]],
    logger: logging.Logger,
) -> pd.DataFrame:
  """Preprocesses categorical columns of the dataframe.

  Args:
    dataframe: A dataframe for the preprocessing.
    column_metadata: A mapping between "categorical" and "numerical" data types
      and column names.
    logger: A logger object to log and save preprocessing messages.

  Returns:
    A dataframe with processed categorical columns.
  """
  categorical_column_names = column_metadata.get("categorical", None)
  if not categorical_column_names:
    logger.info("No categorical columns are specified for the dataframe.")
    return dataframe
  categorical_dataframe = dataframe[categorical_column_names]
  for (
      categorical_column_name,
      categorical_column_data,
  ) in categorical_dataframe.items():
    check_categorical_column(
        categorical_column_name=categorical_column_name,
        categorical_column_data=categorical_column_data,
        logger=logger,
    )
  return dataframe
