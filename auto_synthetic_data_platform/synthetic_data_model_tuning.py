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

"""A module with an end-to-end synthetic data model training."""
import datetime
import functools
import logging
import pathlib
from typing import Any, Final, Literal, Mapping, Sequence, TypeVar
from auto_synthetic_data_platform import experiment_logging
from auto_synthetic_data_platform import custom_constraints
import optuna
from optuna import study
from optuna import visualization
import pandas as pd
from pandas.io.formats import style
from synthcity import benchmark
from synthcity import plugins
from synthcity import utils
from synthcity.plugins.core import dataloader
from synthcity.utils import optuna_sample

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
_REQUIRED_EVALUATION_COLUMNS: Final[list] = ["mean", "direction"]


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
    ).dataframe()
  return model.generate(count=count, random_state=random_state).dataframe()


def list_relevant_evaluation_row_names(
    *, evaluation_category: Sequence[str], evaluation_metrics: Sequence[str]
) -> Sequence[str]:
  """Returns a sequence with relevant evaluation metric names.

  Args:
    evaluation_category: A sequence with an evaluation category name.
    evaluation_metrics: A sequence with an evaluation metric name.
  """
  return [
      ".".join([category, metric])
      for category, metric in zip(
          [evaluation_category] * len(evaluation_metrics), evaluation_metrics
      )
  ]


def verify_column_names(
    *,
    evaluation_results_mapping: Sequence[Mapping[str, pd.DataFrame]],
) -> None:
  """Verifies that all evaluation results contain the required column names.

  Raises:
    ValueError: An error if an evaluation results doesn't contain both
    "mean" and "direction" columns.
  """
  for model_name, evaluation_scores in evaluation_results_mapping.items():
    actual_columns = evaluation_scores.columns.to_list()
    if not set(_REQUIRED_EVALUATION_COLUMNS).issubset(actual_columns):
      missing_columns = list(
          set(_REQUIRED_EVALUATION_COLUMNS) - set(actual_columns)
      )
      raise ValueError(
          f"The evaluation results for model {model_name!r} miss the required"
          f" column(s): {', '.join(missing_columns)}. It must contain:"
          f" {', '.join(_REQUIRED_EVALUATION_COLUMNS)}."
      )


def verify_indexes(
    *,
    evaluation_results_mapping: Sequence[Mapping[str, pd.DataFrame]],
) -> None:
  """Verifies if all evaluation results contain the same indexes.

  Raises:
    ValueError: An error if any of the evaluation has indexes different than
    the rest.
  """
  indexes = [
      evaluation_results.index.to_list()
      for evaluation_results in evaluation_results_mapping.values()
  ]
  if not indexes[1:] == indexes[:-1]:
    raise ValueError(
        f"Not all evaluation results have identicial indexes (evaluation"
        f" tests)."
    )


def setup_highlights(
    row: pd.Series, *, row_directions: Mapping[str, str]
) -> Sequence[str]:
  """Returns a sequence of dataframe formating rules for a given row.

  Args:
    row: A series with evaluation results.
    row_direction: A mapping between all evaluation test names and the direction
      in which they should be optimized.
  """
  if row_directions.get(row.name) == _MINIMIZE:
    best_evaluation_score = row.min()
    worst_evaluation_score = row.max()
  else:
    best_evaluation_score = row.max()
    worst_evaluation_score = row.min()
  styles = []
  for evaluation_score in row.values:
    if evaluation_score == best_evaluation_score:
      styles.append(_BEST_EVALUATION_SCORE_HIGHLIGHT)
    elif evaluation_score == worst_evaluation_score:
      styles.append(_WORST_EVALUATION_SCORE_HIGHLIGHT)
    else:
      styles.append(_DEFAULT_EVALUATION_SCORE_HIGHLIGHT)
  return styles


def compare_synthetic_data_models_full_evaluation_reports(
    *,
    evaluation_results_mapping: Sequence[Mapping[str, pd.DataFrame]],
) -> style.Styler:
  """Returns a styled dataframe with highlighted best and worst results.

  Args:
    evaluation_results_mapping:  A mapping between model names and respective
      full evalaution reports.

  Raises:
    ValueError: An error if an evaluation results doesn't contain both
    "mean" and "direction" columns. Or if any of the evaluation has results indexes
    different than the rest.
  """
  verify_column_names(evaluation_results_mapping=evaluation_results_mapping)
  verify_indexes(evaluation_results_mapping=evaluation_results_mapping)
  evaluation_mean_scores = [
      scores["mean"] for scores in evaluation_results_mapping.values()
  ]
  directions = evaluation_results_mapping[
      next(iter(evaluation_results_mapping))
  ]["direction"].to_dict()
  output_dataframe = pd.concat(evaluation_mean_scores, axis=1)
  output_dataframe.set_axis(
      evaluation_results_mapping.keys(), axis=1, inplace=True
  )
  return output_dataframe.style.apply(
      functools.partial(setup_highlights, row_directions=directions), axis=1
  )


class SyntheticDataModelTuner:
  """Trains a synthetic data model using the most optimal hyperparameters.

  Attributes:
    train_data_loader: A 'synthcity' data loader initialized with a dataframe
      containing real preprocessed train data.
    test_data_loader: A 'synthcity' data loader initialized with a dataframe
      containing real preprocessed test data.
    synthetic_data_model: A 'synthcity' synthetic data model to optimize.
    task_type: A task type compatible with the 'synthcity' library. Used for
      model evaluation.
    experiment_directory: A path to the experiment directory where all the
      experiment artifacts will be saved.
    number_of_trials: A number of hyperoptimization trials to run.
    optimization_direction: Optimization direction of evaluation metrics. One of
      'minimize' or 'maximize'.
    evaluation_metrics: A sequence of metrics to test.
    evaluate_kwargs: A mapping of keywords arguments for the 'synthcity'
      Benchmarks.evaluate method and their values.
    logger: A logger object to log and save hyperparameter optimization
      messages.
    study: An 'optuna' hyperarameter optimization study.
    best_hyperparameters: A mapping of the most optimal hyperparameter names and
      their values.
    best_synthetic_data_model: An optimized 'synthcity' synthetic data model.
    best_synthetic_data_model_full_evaluation_report: A dataframe with all available
      evaluation results of the best model.
    best_synthetic_data_model_evaluation_report: A dataframe with only
      'evaluation_metrics' results of the best model.
  """

  def __init__(
      self,
      *,
      data_loader: _BaseLoaderType,
      synthetic_data_model: _BaseModelType,
      task_type: Literal[
          "classification",
          "regression",
          "survival_analysis",
          "time_series",
          "time_series_survival",
      ],
      number_of_trials: int,
      optimization_direction: Literal["minimize", "maximize"],
      evaluation_metrics: Mapping[str, list[str]],
      experiment_directory: pathlib.Path | None = None,
      evaluate_kwargs: Mapping[str, Any] | None = None,
  ) -> None:
    """Initializes the SyntheticDataModelOptimizer class.

    Args:
      data_loader: A 'synthcity' data loader initialized with a dataframe
        containing all of the real preprocessed data.
      synthetic_data_model: A 'synthcity' synthetic data model.
      task_type: A task type compatible with the 'synthcity' library. Used for
        model evaluation.
      number_of_trials: A number of hyperoptimization trials to run.
      optimization_direction: Optimization direction of evaluation metrics. One
        of 'minimize' or 'maximize'.
      evaluation_metrics: List of metrics to test. If None, all metrics are
        evaluated. Full dictionary of metrics is: { 'sanity': ['data_mismatch',
        'common_rows_proportion', 'nearest_syn_neighbor_distance',
        'close_values_probability', 'distant_values_probability'],
        'stats':['jensenshannon_dist', 'chi_squared_test', 'feature_corr',
        'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy',
        'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
        'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
        'detection': ['detection_xgb', 'detection_mlp', 'detection_gmm',
        'detection_linear'], 'privacy': ['delta-presence', 'k-anonymization',
        'k-map', 'distinct l-diversity', 'identifiability_score',
        'DomiasMIA_BNAF', 'DomiasMIA_KDE', 'DomiasMIA_prior'] }
      experiment_directory: A path to the experiment directory where all the
        experiment artifacts will be saved.
      evaluate_kwargs: A mapping of keywords arguments for the 'synthcity'
        Benchmarks.evaluate method and their values.
    """
    (
        self.train_data_loader,
        self.test_data_loader,
    ) = split_data_loader_into_train_test(data_loader=data_loader)
    self.synthetic_data_model = synthetic_data_model
    self.task_type = verify_task_type(task_type=task_type)
    self.number_of_trials = number_of_trials
    self.optimization_direction = optimization_direction
    self.evaluation_metrics = evaluation_metrics
    self.experiment_directory = verify_experiment_directory(
        experiment_directory=experiment_directory
    )
    self.evaluate_kwargs = evaluate_kwargs

    verify_optimization_direction_and_evaluation_metrics(
        optimization_direction=self.optimization_direction,
        selected_evaluation_metrics=self.evaluation_metrics,
    )

  @functools.cached_property
  def logger(self) -> logging.Logger:
    """Returns an object for logging hyperparameter optimization messages."""
    return experiment_logging.setup_logger(
        experiment_directory=self.experiment_directory,
        logger_name="hyperparameter_optimization",
    )

  def optimization_function(
      self,
      trial: optuna.Trial,
  ) -> float:
    """Defines a synthetic data model hyperprameter optimization process.

    Args:
      trial: The 'optuna' trial process of evaluating an objective function.

    Returns:
      A mean evaluation metric score.
    """
    trial_id = f"trial_{trial.number}"
    hyperparameter_space = self.synthetic_data_model.hyperparameter_space()
    trial_parameters = optuna_sample.suggest_all(trial, hyperparameter_space)
    self.logger.info(
        "Trial: %s. Model: %s. Selected hyperparameters: %s"
        % (
            trial_id,
            self.synthetic_data_model.name(),
            trial_parameters,
        )
    )
    try:
      if self.evaluate_kwargs:
        report = benchmark.Benchmarks.evaluate(
            [(trial_id, self.synthetic_data_model.name(), trial_parameters)],
            self.train_data_loader,
            task_type=self.task_type,
            repeats=1,
            metrics=self.evaluation_metrics,
            workspace=self.experiment_directory,
            **self.evaluate_kwargs,
        )
      else:
        report = benchmark.Benchmarks.evaluate(
            [(trial_id, self.synthetic_data_model.name(), trial_parameters)],
            self.train_data_loader,
            task_type=self.task_type,
            repeats=1,
            metrics=self.evaluation_metrics,
            workspace=self.experiment_directory,
        )
    except Exception as e:
      self.logger.info(
          "Trial: %s. Error: %s"
          % (
              trial_id,
              type(e).__name__,
          )
      )
      raise optuna.TrialPruned()
    return (
        report[trial_id]
        .query(f'direction == "{self.optimization_direction}"')["mean"]
        .mean()
    )

  @functools.cached_property
  def study(self) -> study.Study:
    """Returns the 'optuna' study after hyperparameter optimization."""
    self.logger.info(
        "Specified 'evaluation_metrics': %s. Optimization direction: '%s'."
        % (self.evaluation_metrics, self.optimization_direction)
    )
    study = optuna.create_study(direction=self.optimization_direction)
    study.optimize(self.optimization_function, n_trials=self.number_of_trials)
    return study

  @functools.cached_property
  def best_hyperparameters(self) -> Mapping[str, Any]:
    """Returns a set of optimized hyperparameters."""
    return self.study.best_params

  def display_parallel_hyperparameter_coordinates(self):
    """Displays a plot of hyperparameter parallel coordinates."""
    return visualization.plot_parallel_coordinate(self.study)

  def display_hyperparameter_importances(self):
    """Displays a plot of hyperparameter parallel coordinates."""
    return visualization.plot_param_importances(self.study)

  @functools.cached_property
  def best_synthetic_data_model(self) -> _BaseModelType:
    """Returns a synthetic data model trained using the best hyperparameters."""
    self.logger.info(
        "Model: %s. Best hyperparameters: %s"
        % (
            self.synthetic_data_model.name(),
            self.best_hyperparameters,
        )
    )
    best_synthetic_data_model = plugins.Plugins().get(
        self.synthetic_data_model.name(),
        **self.best_hyperparameters,
        workspace=self.experiment_directory,
    )
    return best_synthetic_data_model.fit(self.train_data_loader)

  @functools.cached_property
  def best_synthetic_data_model_full_evaluation_report(
      self,
  ) -> pd.DataFrame:
    """Returns a dataframe with all available evaluation results of the best model."""
    if self.evaluate_kwargs:
      return benchmark.Benchmarks.evaluate(
          [(
              "evaluation",
              self.synthetic_data_model.name(),
              self.best_hyperparameters,
          )],
          self.train_data_loader,
          self.test_data_loader,
          task_type=self.task_type,
          repeats=1,
          workspace=self.experiment_directory,
          **self.evaluate_kwargs,
      )
    return benchmark.Benchmarks.evaluate(
        [(
            "evaluation",
            self.synthetic_data_model.name(),
            self.best_hyperparameters,
        )],
        self.train_data_loader,
        self.test_data_loader,
        task_type=self.task_type,
        repeats=1,
        workspace=self.experiment_directory,
    )["evaluation"]

  @functools.cached_property
  def best_synthetic_data_model_evaluation_report(
      self,
  ) -> pd.DataFrame:
    """Returns a dataframe with only 'evaluation_metrics' results of the best model."""
    full_evaluation_report = self.best_synthetic_data_model_full_evaluation_report
    relevant_evaluation_row_names_mapping = {
        evaluation_category: list_relevant_evaluation_row_names(
            evaluation_category=evaluation_category,
            evaluation_metrics=evaluation_metrics,
        )
        for evaluation_category, evaluation_metrics in self.evaluation_metrics.items()
    }
    relevant_evaluation_row_names = [
        sublist
        for sublists in relevant_evaluation_row_names_mapping.values()
        for sublist in sublists
    ]
    return full_evaluation_report[
        full_evaluation_report.index.str.startswith(
            tuple(relevant_evaluation_row_names)
        )
    ]

  def save_best_synthetic_data_model(
      self, *, model_path: pathlib.Path | None = None
  ) -> None:
    """Saves the best synthetic data model to a file."""
    if not model_path:
      creation_time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
      model_name = self.synthetic_data_model.name() + creation_time + ".pkl"
      model_path = self.experiment_directory.joinpath(
          "synthetic_data_models"
      ).joinpath(model_name)
    if not model_path.parent.exists():
      model_path.parent.mkdir()
    utils.serialization.save_to_file(
        str(model_path), self.best_synthetic_data_model
    )

  def generate_synthetic_data_with_the_best_synthetic_data_model(
      self,
      *,
      count: int,
      random_state: int | None = 0,
      custom_constraint: custom_constraints.CustomConstraints | None = None,
      generate_kwargs: Mapping[str, Any] | None = None,
  ) -> pd.DataFrame:
    """Returns a dataframe with synthetic data.

    Args:
      custom_constraint: Additional constraints of type CustomConstraints not
        covered within regular 'synthcity constraints' spec.
      count: A number of observations to create.
      random_state: A random state to ensure results repeatability.
        self.best_synthetic_data_model will be used.
      evaluate_kwargs: A mapping of keywords arguments for the 'synthcity'
        Plugin.generate method and their values.
    """
    custom_constrained_synth_data = pd.DataFrame()

    # generate custom constrained synthetic data
    if custom_constraint:
      for it in range(custom_constraint.custom_sampling_patience):
        synth_data = generate_synthetic_data_with_synthetic_data_model(
            count=count,
            model=self.best_synthetic_data_model,
            random_state=random_state,
            generate_kwargs=generate_kwargs,
        )
        synth_data = synth_data[custom_constraint.validity_func(synth_data)]
        custom_constrained_synth_data = pd.concat(
            [custom_constrained_synth_data, synth_data], ignore_index=True
        ).reset_index(drop=True)
        if (
            len(custom_constrained_synth_data)
            >= custom_constraint.custom_constraint_count
        ):
          custom_constrained_synth_data = custom_constrained_synth_data.iloc[
              : custom_constraint.custom_constraint_count, :
          ]
          break

    # generate other constrained synthetic data
    other_synth_data = generate_synthetic_data_with_synthetic_data_model(
        count=count,
        model=self.best_synthetic_data_model,
        random_state=random_state,
        generate_kwargs=generate_kwargs,
    )
    return (
        pd.concat(
            [custom_constrained_synth_data, other_synth_data], ignore_index=True
        )
        .sample(frac=1)
        .reset_index(drop=True)
    )