import datetime
import pathlib
from typing import Any, Final, Literal, Mapping, TypeVar
import pandas as pd
from synthcity import plugins
from synthcity import utils
from synthcity.plugins.core import dataloader

_MINIMIZE: Final[str] = "minimize"
_MAXIMIZE: Final[str] = "maximize"
_AVAILABLE_OPTIMIZATION_DIRECTIONS = (_MINIMIZE, _MAXIMIZE)
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
_MINIMIZE_EVALUATION_METRICS: Final[dict] = {
    "sanity": [
        "data_mismatch",
        "common_rows_proportion",
        "nearest_syn_neighbor_distance",
        "distant_values_probability",
    ],
    "stats": [
        "jensenshannon_dist",
        "max_mean_discrepancy",
        "wasserstein_dist",
    ],
    "detection": [
        "detection_xgb",
        "detection_mlp",
        "detection_gmm",
        "detection_linear",
    ],
    "privacy": [
        "identifiability_score",
        "DomiasMIA_BNAF",
        "DomiasMIA_KDE",
        "DomiasMIA_prior",
    ],
}
_MAXIMIZE_EVALUATION_METRICS: Final[dict] = {
    "sanity": [
        "close_values_probability",
    ],
    "stats": [
        "chi_squared_test",
        "inv_kl_divergence",
        "ks_test",
        "prdc",
        "alpha_precision",
    ],
    "performance": ["linear_model", "mlp", "xgb", "feat_rank_distance"],
    "privacy": [
        "delta-presence",
        "k-anonymization",
        "k-map",
        "distinct l-diversity",
    ],
}
_BEST_EVALUATION_SCORE_HIGHLIGHT: Final[str] = "background-color: green;"
_WORST_EVALUATION_SCORE_HIGHLIGHT: Final[str] = "background-color: red;"
_DEFAULT_EVALUATION_SCORE_HIGHLIGHT: Final[str] = ""
_BaseLoaderType = TypeVar("_BaseLoaderType", bound=dataloader.DataLoader)
_BaseModelType = TypeVar("_BaseModelType", bound=plugins.Plugin)


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


def verify_optimization_direction_and_evaluation_metrics(
    *,
    optimization_direction: Literal["minimize", "maximize"],
    selected_evaluation_metrics: Mapping[str, list[str]],
) -> None:
  """Verifies if the specified optimization direction is compatible.

  Args:
    optimization_direction: Optimization direction of evaluation metrics. One of
      'minimize' or 'maximize'.
    selected_evaluation_metrics: A mapping between 'selected' evaluation
      categories and sequences of their respective evaluation metric names.

  Raises:
      ValueError: An error if the specified optimization direction is not
      compatible or either category or evaluation metric are not recognized.
  """
  if optimization_direction not in _AVAILABLE_OPTIMIZATION_DIRECTIONS:
    raise ValueError(
        f"{optimization_direction!r} not compatible. Choose one from:"
        f" {', '.join(_AVAILABLE_OPTIMIZATION_DIRECTIONS)}"
    )

  if optimization_direction == _MINIMIZE:
    verify_evaluation_metrics(
        selected_evaluation_metrics=selected_evaluation_metrics,
        reference_evaluation_metrics=_MINIMIZE_EVALUATION_METRICS,
    )
  else:
    verify_evaluation_metrics(
        selected_evaluation_metrics=selected_evaluation_metrics,
        reference_evaluation_metrics=_MAXIMIZE_EVALUATION_METRICS,
    )


def verify_experiment_directory(
    *, experiment_directory: pathlib.Path | None
) -> pathlib.Path:
  """Returns an experiment directory where all artifacts will be saved.

  Args:
    experiment_directory: A path to a directory where all artifacts should be
      saved.
  """
  if not experiment_directory:
    creation_time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    experiment_directory = pathlib.Path.cwd().joinpath(creation_time)
  if not experiment_directory.exists():
    experiment_directory.mkdir()
  return experiment_directory


def generate_synthetic_data_with_synthetic_data_model(
    *,
    count: int,
    model: pathlib.Path | _BaseModelType,
    random_state: int | None = 0,
    generate_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
  """Returns a dataframe with synthetic data.

  Args:
    count: A number of observations to create.
    model: The path to the best synthetic data model or its instance.
    random_state: A random state to ensure results repeatability.
      self.best_synthetic_data_model will be used.
    evaluate_kwargs: A mapping of keywords arguments for the 'synthcity'
      Plugin.generate method and their values.
  """
  if isinstance(model, pathlib.Path):
    model = utils.serialization.load_from_file(model)
  if generate_kwargs:
    return model.generate(
        count=count, random_state=random_state, **generate_kwargs
    )
  return model.generate(count=count, random_state=random_state).dataframe()
