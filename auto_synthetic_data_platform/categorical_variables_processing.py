"""A categorical values processing module of the Google EMEA gPS Data Science Auto Synthetic Data Platform."""
import logging
from typing import Final
import pandas as pd

_MILD_IMBALANCE_UPPER_LIMIT: Final[float] = 0.4
_MILD_IMBALANCE_LOWER_LIMIT: Final[float] = 0.2
_MODERATE_IMBALANCE_LOWER_LIMIT: Final[float] = 0.01


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
