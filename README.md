# Auto Synthetic Data Platform
### Auto Synthetic Data Platform is a Python package that will help advertisers create a synthetic version of their first-party datasets (1PD) using AI/ML.
##### This is not an official Google product.

[Introduction](#introduction) •
[Limitations](#limitations) •
[Getting started](#getting-started) •
[References](#references)

## Introduction

Organizations increasingly rely on first-party data (1PD) due to government regulations, depreciation and fragmentation of AdTech solutions, e.g. removal of third-party cookies (3PC). However, typically advertisers are not allowed to share 1PD externally. It blocks leveraging third-party Data Science expertise in new AI/ML model development and limits realizing the full revenue-generating potential of owned 1PD.

The package is a self-served solution to help advertisers create a synthetic version of their first-party datasets (1PD). The solution automatically preprocesses a real dataset and identifies the most optimal synthetic data model through the process of hyperparameter tuning.

The solution will optimize the model against a set of objectives specified by a user (see [Getting started](#getting-started)). Typically users would optimize their model to generate synthetic data of:

- Statistical properties as close as possible to the one of the real data
- High privacy standards to protect sensitivity of real data
- High suitability, i.e. synthetic data could be used to train AI/ML models that later can make accurate predictions on real data

The solution relies on the [SynthCity](https://github.com/vanderschaarlab/synthcity) library developed by The van der Schaar Lab at the University of Cambridge.

## Limitations

Users take full responsibility for synthetic data created using this solution.

The solution can protect user privacy in the case where each row in the original dataset is a user. However, it does not protect, for example, leaking sensitive company data like total revenue per day, since by design it maintains the statistical resemblance of the original data.

## Getting started

### Installation

The *torchaudio* and *torchdata* packages must be uninstalled if you already have PyTorch >= 2.0. It is to make the enviornment compatible with the *synthcity* package.

The *synthcity* team is already working on upgrading their requirements to PyTorch 2.0. See [here](https://github.com/vanderschaarlab/synthcity/issues/234).

```bash
pip -q uninstall torchaudio torchdata
```

The *auto_synthetic_data_platform* package can be installed from a file.

```bash
pip install auto_synthetic_data_platform.tar.gz
```

### Preprocessing & loading data

#### Preprocessing

The *auto_synthetic_data_platform* has the *Preprocessor* class that can help preprocess the real dataset according to the industry best practices. It will also log all the information about the dataset that can impact a synthetic data model training. The logs can be shared externally in your organization cannot allow external people to have access to your data and remote synthetic data model training debugging is required.

First, load your real data to a Pandas dataframe.

```python
import pandas as pd
_REAL_DATAFRAME_PATH = "/content/real_dataset.csv"
real_dataframe = pd.read_csv(_REAL_DATAFRAME_PATH)
```

Second, specify column and preprocessing metadata.

```python
_EXPERIMENT_DIRECTORY = ("/content/")
_COLUMN_METADATA = {
    "numerical": [
        "column_a",
        "column_b",
    ],
    "categorical": [
        "column_c",
        "column_d",
    ],
}
_PREPROCESS_METADATA = {
    "remove_numerical_outliers": False,
    "remove_duplicates": True,
    "preprocess_missing_values": True,
    "missing_values_method": "drop",
}
```

Third, initialize the *Preprocessor* class and preprocess the data.

```python
from auto_synthetic_data_platform import preprocessing
import pathlib
preprocessor = preprocessing.Preprocessor(
    dataframe_path=pathlib.Path(_REAL_DATAFRAME_PATH),
    experiment_directory=pathlib.Path(_EXPERIMENT_DIRECTORY),
    column_metadata=_COLUMN_METADATA,
    preprocess_metadata=_PREPROCESS_METADATA,
)
preprocessed_real_dataset = preprocessor.output_dataframe
```

#### Loading

Lastly, load the preprocessed real dataset using a dataloader from the *SynthCity* package.

```python
from synthcity.plugins.core import dataloader
data_loader = dataloader.GenericDataLoader(
    data=preprocessed_real_dataset,
    target_column="column_d",
)
```

### Synthetic data model training

#### Model selection

At this stage a synthetic data model instance from *SynthCity* needs to be initialized. The list of all the available models can be found [here](https://github.com/vanderschaarlab/synthcity#-methods).

```python
from synthcity import plugins
_TVAE_MODEL = "tvae"
tvae_synthetic_data_model = plugins.Plugins().get(_TVAE_MODEL)
```

#### Objective/s selection

Then, a mapping of evaluation metrics compatible with *SynthCity* is required to proceed further. Each model in the hyperparameter tuning process will be evaluated against these criteria to identify the best synthetic data model. The list of available metrics can be found [here](https://github.com/vanderschaarlab/synthcity#zap-evaluation-metrics).

```python
_EVALUATION_METRICS = {
    "sanity": ["close_values_probability"],
    "stats": ["inv_kl_divergence"],
    "performance": ["xgb"],
    "privacy": ["k-anonymization"],
}
```

#### Tuner setup

A synthetic data model tuner from the *auto_synthetic_data_platform* package needs to be setup.

```python
from auto_synthetic_data_platform import synthetic_data_model_tuning
_NUMBER_OF_TRIALS = 2
_OPTIMIZATION_DIRECTION = "maximize"
_TASK_TYPE = "regression"
tvae_synthetic_data_model_optimizer = synthetic_data_model_tuning.SyntheticDataModelTuner(
    data_loader=data_loader,
    synthetic_data_model=tvae_synthetic_data_model,
    experiment_directory=pathlib.Path(_EXPERIMENT_DIRECTORY),
    number_of_trials=_NUMBER_OF_TRIALS,
    optimization_direction=_OPTIMIZATION_DIRECTION,
    evaluation_metrics=_EVALUATION_METRICS,
    task_type=_TASK_TYPE,
)
```

#### Hyperparameter tuning

The tuner runs a hyperparameter search in the background and then outputs the best synthetic data model. The processing time depends on the number of trials.

**WARNING:** The process is resource & time consuming.

```python
best_tvae_network_synthetic_data_model = (
    tvae_synthetic_data_model_optimizer.best_synthetic_data_model
    )
```

The tuner class provides an easy access to:

- The most optimal hyperparameters for the given model.

```python
best_hyperparameters = tvae_synthetic_data_model_optimizer.best_hyperparameters
```

- A plot with parallel hyperparamter coordinates for later analysis.

```python
tvae_synthetic_data_model_optimizer.display_parallel_hyperparameter_coordinates()
```

- A plot with hyperparamter importances during training.

```python
tvae_synthetic_data_model_optimizer.display_hyperparameter_importances()
```

- A full evaluation report on the idenitifed best synthetic data model using all the available evaluation metrics from *SynthCity*.

```python
best_tvae_model_full_evaluation_report = tvae_synthetic_data_model_optimizer.best_synthetic_data_model_full_evaluation_report
```

- An evaluation report on the idenitifed best synthetic data model using only the evaluation metrics specified at the class initializaiton.

```python
best_tvae_model_evaluation_report = tvae_synthetic_data_model_optimizer.best_synthetic_data_model_evaluation_report
```

The most topimal synthetic data can be easily saved using the tuner class.

```python
tvae_synthetic_data_model_optimizer.save_best_synthetic_data_model()
```

### Synthetic data generation

Synthetic data can be easily genrated with the tuner.

```python
synthetic_data_10_examples = tvae_synthetic_data_model_optimizer.generate_synthetic_data_with_the_best_synthetic_data_model(
    count=10,
)
```

### Comparing multiple tuned synthetic data models

Another likely step is to compare between 2 or more tuned synthetic data models and the *auto_synthetic_data_platform* package can help with it as well.

For the demo purposes, we need to tune an alternative synthetic data model following the exact same steps as above.

```python
_CTGAN_MODEL = "ctgan"
ctgan_synthetic_data_model = plugins.Plugins().get(_CTGAN_MODEL)
ctgan_synthetic_data_model_optimizer = synthetic_data_model_tuning.SyntheticDataModelTuner(
    data_loader=data_loader,
    synthetic_data_model=ctgan_synthetic_data_model,
    experiment_directory=pathlib.Path(_EXPERIMENT_DIRECTORY),
    number_of_trials=_NUMBER_OF_TRIALS,
    optimization_direction=_OPTIMIZATION_DIRECTION,
    evaluation_metrics=_EVALUATION_METRICS,
    task_type=_TASK_TYPE,
)
best_ctgan_synthetic_data_model = (
    ctgan_synthetic_data_model_optimizer.best_synthetic_data_model
    )
best_ctgan_model_evaluation_report = ctgan_synthetic_data_model_optimizer.best_synthetic_data_model_evaluation_report
```

The last step is to compare 2 or more created evaluation reports.

```python
evaluation_reports_to_compare = {
    "tvae": best_tvae_model_evaluation_report,
    "ctgan": best_ctgan_model_evaluation_report,
}
synthetic_data_model_tuning.compare_synthetic_data_models_full_evaluation_reports(
    evaluation_results_mapping=evaluation_reports_to_compare
)
```

## References

The *auto_synthetic_data_platform* package is a wrapper around the *synthcity* library. Credits to: Qian, Zhaozhi and Cebere, Bogdan-Constantin and van der Schaar, Mihaela

The *auto_synthetic_data_platform* package uses the Apache License (Version 2.0, January 2004) more details in the package's LICENSE file.
