from typing import Final, Literal, Mapping, TypeVar
from synthcity.plugins.core import dataloader

_MINIMIZE: Final[str] = "minimize"
_MAXIMIZE: Final[str] = "maximize"
_CLASSIFICATION: Final[str] = "classification"
_REGRESSION: Final[str] = "regression"
_SURVIVAL_ANALYSIS: Final[str] = "survival_analysis"
_TIME_SERIES: Final[str] = "time_series"
_TIME_SERIES_SURVIVAL: Final[str] = "time_series_survival"
_AVAILABLE_TASK_TYPES = (
    _CLASSIFICATION,
    _REGRESSION,
    _SURVIVAL_ANALYSIS,
    _TIME_SERIES,
    _TIME_SERIES_SURVIVAL,
)
_BaseLoaderType = TypeVar("_BaseLoaderType", bound=dataloader.DataLoader)


def split_data_loader_into_train_test(
    *, data_loader: _BaseLoaderType
) -> tuple[_BaseLoaderType, _BaseLoaderType]:
  """Returns train and test 'synthcity' data loaders.

  Args:
      data_loader: A 'synthcity' data loader initialized with a dataframe
        containing all of the real preprocessed data.
  """
  return data_loader.train(), data_loader.test()


def verify_task_type(
    *,
    task_type: Literal[
        "classification",
        "regression",
        "survival_analysis",
        "time_series",
        "time_series_survival",
    ],
) -> str:
  """Returns a task type for model evaluation.

  Args:
      task_type: A task type compatible with the 'synthcity' library. Used for
        model evaluation.

  Raises:
      ValueError: An error if the provided task type is not compatible with the
      'synthcity' library.
  """
  if task_type not in _AVAILABLE_TASK_TYPES:
    raise ValueError(
        f"{task_type!r} task type is not compatible. It must be one of:"
        f" {', '.join(_AVAILABLE_TASK_TYPES)}."
    )
  return task_type


def verify_evaluation_metrics(
    *,
    selected_evaluation_metrics: Mapping[str, list[str]],
    reference_evaluation_metrics: Mapping[str, list[str]],
) -> None:
  """Verifies if the specified evaluation metrics are compatible.

  Args:
    selected_evaluation_metrics: A mapping between 'selected' evaluation
      categories and sequences of their respective evaluation metric names.
    selected_evaluation_metrics: A mapping between reference evaluation
      categories and sequences of their respective evaluation metric names.

  Raises:
    ValueError: An error if an evaluation category or an evaluation metric
    is not recognized.
  """
  reference_evaluation_categories = reference_evaluation_metrics.keys()
  for evaluation_category in selected_evaluation_metrics.keys():
    if evaluation_category not in reference_evaluation_categories:
      raise ValueError(
          f"{evaluation_category!r} evaluation category is not recognized. "
          f"It must be one of: {', '.join(reference_evaluation_categories)}."
      )
    category_reference_evaluation_metrics = reference_evaluation_metrics[
        evaluation_category
    ]
    for evaluation_metric in selected_evaluation_metrics[evaluation_category]:
      if evaluation_metric not in category_reference_evaluation_metrics:
        raise ValueError(
            f"{evaluation_metric!r} evaluation metric is not recognized. "
            f"It must be one of: {', '.join(reference_evaluation_metrics)}."
        )
