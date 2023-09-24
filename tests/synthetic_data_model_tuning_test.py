"""Tests for utility functions in synthetic_data_model_tuning.py."""
from typing import Final
from absl.testing import parameterized
from auto_synthetic_data_platform import synthetic_data_model_tuning
import pandas as pd
from synthcity.plugins.core import dataloader

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
