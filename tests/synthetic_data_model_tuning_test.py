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

"""Tests for utility functions in synthetic_data_model_tuning.py."""
import logging
import pathlib
import tempfile
from typing import Final
from absl.testing import absltest
from absl.testing import parameterized
from auto_synthetic_data_platform import synthetic_data_model_tuning
from auto_synthetic_data_platform import custom_constraints
import optuna
import pandas as pd
from synthcity import plugins
from synthcity.plugins.core import dataloader
from synthcity.plugins.generic import plugin_dummy_sampler

_DATAFRAME = pd.DataFrame.from_dict({
    "categorical_column": [0, 1, 3, 4, 5, 6],
    "numerical_column": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
})
_METRICS: Final[dict] = {
    "sanity": [
        "data_mismatch",
        "common_rows_proportion",
        "nearest_syn_neighbor_distance",
        "distant_values_probability",
    ],
}
_DATA_LOADER = dataloader.GenericDataLoader(
    _DATAFRAME, train_size=0.5, target_column="numerical_column"
)
_NUMBER_OF_TRIALS: Final[int] = 1
_OPTIMIZATION_DIRECTION: Final[str] = "minimize"
_EVALUATION_METRICS: Final[dict] = {"sanity": ["common_rows_proportion"]}
_COUNT: Final[int] = 1


class SyntheticDataModelTuningTests(parameterized.TestCase):

  def test_split_data_loader_into_train_test(self):
    (
        train_data_loader,
        test_data_loader,
    ) = synthetic_data_model_tuning.split_data_loader_into_train_test(
        data_loader=_DATA_LOADER
    )
    self.assertTrue(
        all([
            isinstance(train_data_loader, dataloader.DataLoader),
            isinstance(test_data_loader, dataloader.DataLoader),
        ])
    )

  def test_verify_task_type(self):
    with self.assertRaisesRegex(ValueError, "task type is not compatible"):
      synthetic_data_model_tuning.verify_task_type(
          task_type="incompatible_task_type",
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="evaluation_category",
          selected_evaluation_metrics=dict(
              incomatible_evaluation_category="mse"
          ),
          error_message="evaluation category is not recognized.",
      ),
      dict(
          testcase_name="evaluation_metric",
          selected_evaluation_metrics=dict(sanity="mse"),
          error_message="evaluation metric is not recognized.",
      ),
  )
  def test_verify_evaluation_metrics(
      self, selected_evaluation_metrics, error_message
  ):
    with self.assertRaisesRegex(ValueError, error_message):
      synthetic_data_model_tuning.verify_evaluation_metrics(
          selected_evaluation_metrics=selected_evaluation_metrics,
          reference_evaluation_metrics=_METRICS,
      )

  def test_verify_optimization_direction_and_evaluation_metrics_value_error(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError, "'incompatible_optimization_metric' not compatible"
    ):
      synthetic_data_model_tuning.verify_optimization_direction_and_evaluation_metrics(
          optimization_direction="incompatible_optimization_metric",
          selected_evaluation_metrics=_METRICS,
      )

  def test_verify_experiment_directory(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory).joinpath(
          "test_directory"
      )
      actual_output = synthetic_data_model_tuning.verify_experiment_directory(
          experiment_directory=experiment_directory
      )
      self.assertEqual(str(actual_output), str(experiment_directory))

  def test_generate_synthetic_data_with_the_best_synthetic_data_model(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
      )
      actual_output = synthetic_data_model_tuning.generate_synthetic_data_with_synthetic_data_model(
          count=_COUNT, model=tuner.best_synthetic_data_model
      )
      self.assertEqual(actual_output["categorical_column"].loc[0], 4)

  def test_generate_synthetic_data_with_the_best_synthetic_data_model_from_path(
      self,
  ):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
      )
      model_path = pathlib.Path(temporary_directory).joinpath("test_model.pkl")
      tuner.save_best_synthetic_data_model(model_path=model_path)
      actual_output = synthetic_data_model_tuning.generate_synthetic_data_with_synthetic_data_model(
          count=_COUNT, model=model_path
      )
      self.assertEqual(actual_output["categorical_column"].loc[0], 4)

  def test_list_relevant_evaluation_row_names(self):
    input_metrics = {
        "sanity": ["close_values_probability", "data_mismatch"],
        "stats": ["inv_kl_divergence", "jensenshannon_dist"],
        "performance": ["xgb", "mlp"],
        "privacy": ["k-anonymization", "k-map"],
    }
    actual_output = {
        evaluation_category: (
            synthetic_data_model_tuning.list_relevant_evaluation_row_names(
                evaluation_category=evaluation_category,
                evaluation_metrics=evaluation_metrics,
            )
        )
        for evaluation_category, evaluation_metrics in input_metrics.items()
    }
    expected_output = {
        "sanity": ["sanity.close_values_probability", "sanity.data_mismatch"],
        "stats": ["stats.inv_kl_divergence", "stats.jensenshannon_dist"],
        "performance": ["performance.xgb", "performance.mlp"],
        "privacy": ["privacy.k-anonymization", "privacy.k-map"],
    }
    self.assertEqual(actual_output, expected_output)

  def test_verify_column_names(self):
    evaluation_results_mapping = dict(
        model_name=pd.DataFrame({"mean": [0], "incompatible_column": [0]})
    )
    with self.assertRaisesRegex(ValueError, "miss the required"):
      synthetic_data_model_tuning.verify_column_names(
          evaluation_results_mapping=evaluation_results_mapping,
      )

  def test_verify_indexes(self):
    model_a_dataframe = pd.DataFrame(
        {"mean": [0], "incompatible_column": [0]}, index=["performance"]
    )
    model_b_dataframe = pd.DataFrame(
        {"mean": [0], "incompatible_column": [0]}, index=["stats"]
    )
    evaluation_results_mapping = dict(
        model_a=model_a_dataframe, model_b=model_b_dataframe
    )
    with self.assertRaisesRegex(
        ValueError, "Not all evaluation results have identicial indexes"
    ):
      synthetic_data_model_tuning.verify_indexes(
          evaluation_results_mapping=evaluation_results_mapping,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="minimize",
          row_directions={0: "minimize"},
          row=pd.DataFrame([[0, 1, 2]]).iloc[0],
          expected_output=[
              "background-color: green;",
              "",
              "background-color: red;",
          ],
      ),
      dict(
          testcase_name="maximize",
          row_directions={0: "maximize"},
          row=pd.DataFrame([[0, 1, 2]]).iloc[0],
          expected_output=[
              "background-color: red;",
              "",
              "background-color: green;",
          ],
      ),
  )
  def test_setup_highlights(self, row, row_directions, expected_output):
    actual_output = synthetic_data_model_tuning.setup_highlights(
        row=row, row_directions=row_directions
    )
    self.assertEqual(actual_output, expected_output)


class SyntheticDataModelTunerTests(parameterized.TestCase):

  def _is_valid_element(self, synth_test_df:pd.DataFrame) -> pd.Series:
    #assume element is valid if it contains a value == 4 in categorical column.
    return synth_test_df["categorical_column"]==4

  def test_synthetic_data_model_tuner_logger(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
      )
      self.assertIsInstance(tuner.logger, logging.Logger)

  def test_synthetic_data_model_tuner_optimization_function_without_evaluate_kwargs(
      self,
  ):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
      )
      trial = optuna.trial.create_trial(state=optuna.trial.TrialState.WAITING)
      self.assertEqual(tuner.optimization_function(trial), 0.9999999900000002)

  def test_synthetic_data_model_tuner_optimization_function_with_evaluate_kwargs(
      self,
  ):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
          evaluate_kwargs=dict(augmentation_rule="log"),
      )
      trial = optuna.trial.create_trial(state=optuna.trial.TrialState.WAITING)
      self.assertEqual(tuner.optimization_function(trial), 0.9999999900000002)

  def test_synthetic_data_model_tuner_optimization_function_exception(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
          evaluate_kwargs=dict(task_type=1),
      )
      trial = optuna.trial.create_trial(state=optuna.trial.TrialState.WAITING)
      with self.assertRaises(optuna.exceptions.TrialPruned):
        tuner.optimization_function(trial)

  def test_synthetic_data_model_tuner_study(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
      )
      self.assertAlmostEqual(tuner.study.best_value, 0.9999999900000002)

  def test_synthetic_data_model_tuner_best_hyperparameters(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
      )
      self.assertEqual(tuner.best_hyperparameters, {})

  def test_synthetic_data_model_tuner_best_synthetic_data_model(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
      )
      self.assertIsInstance(
          tuner.best_synthetic_data_model,
          plugin_dummy_sampler.DummySamplerPlugin,
      )

  def test_synthetic_data_model_tuner_best_synthetic_data_model_full_evaluation_report(
      self,
  ):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
      )
      actual_result = (
          tuner.best_synthetic_data_model_full_evaluation_report[
              "mean"
          ].loc["sanity.data_mismatch.score"]
      )
      self.assertEqual(actual_result, 0.0)

  def test_synthetic_data_model_tuner_best_synthetic_data_model_evaluation_report(
      self,
  ):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
      )
      actual_result = (
          tuner.best_synthetic_data_model_evaluation_report[
              "mean"
          ].loc["sanity.common_rows_proportion.score"]
      )
      self.assertEqual(actual_result, 0.9999999900000002)

  def test_synthetic_data_model_tuner_save_best_synthetic_data_model(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
      )
      tuner.save_best_synthetic_data_model()
      files_in_experiment_directory = list(experiment_directory.rglob("*"))
      self.assertTrue(
          any(
              file_name.suffix == ".pkl"
              for file_name in files_in_experiment_directory
          )
      )

  def test_synthetic_data_model_tuner_save_best_synthetic_data_model_from_path(
      self,
  ):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
      )
      model_path = pathlib.Path(temporary_directory).joinpath("test_model.pkl")
      tuner.save_best_synthetic_data_model(model_path=model_path)
      self.assertTrue(model_path.exists())

  def test_generate_synthetic_data_with_the_best_synthetic_data_model(
      self,
  ):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
      )
      actual_output = tuner.generate_synthetic_data_with_the_best_synthetic_data_model(
          count=_COUNT,
      )
      self.assertEqual(actual_output["categorical_column"].loc[0], 4)

  def test_generate_synthetic_data_with_the_best_synthetic_data_model_with_custom_constraints(
    self,
  ):
    with tempfile.TemporaryDirectory() as temporary_directory:
      experiment_directory = pathlib.Path(temporary_directory)
      model = plugins.Plugins().get(
          "dummy_sampler", workspace=experiment_directory
      )
      tuner = synthetic_data_model_tuning.SyntheticDataModelTuner(
          data_loader=_DATA_LOADER,
          synthetic_data_model=model,
          task_type="classification",
          number_of_trials=_NUMBER_OF_TRIALS,
          optimization_direction=_OPTIMIZATION_DIRECTION,
          evaluation_metrics=_EVALUATION_METRICS,
          experiment_directory=experiment_directory,
      )

      custom_constraint = custom_constraints.CustomConstraints(
          self._is_valid_element, custom_sampling_patience=2, count=2
      )
      actual_output = (
          tuner.generate_synthetic_data_with_the_best_synthetic_data_model(
              count=_COUNT, custom_constraint=custom_constraint
          )
      )
      self.assertTrue(list(actual_output["categorical_column"]).count(4) >= 2)

if __name__ == "__main__":
  absltest.main()
