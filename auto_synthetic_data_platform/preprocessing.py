"""An end-to-end real data preprocessing module of the Google EMEA gPS Data Science Auto Synthetic Data Platform."""
import logging
from typing import Any, Final, Mapping
from auto_synthetic_data_platform import categorical_variables_processing
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
