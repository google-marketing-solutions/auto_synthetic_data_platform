"""A process logging module of the Google EMEA gPS Data Science Auto Synthetic Data Platform."""
import logging
import pathlib


def setup_logger(
    *,
    experiment_directory: pathlib.Path,
    logger_name: str,
) -> logging.Logger:
  """Returns a logger object for logging and saving messages.

  Args:
   experiment_directory: A path to a experiment directory.
   logger_name: A logger name.
  """
  log_directory = experiment_directory.joinpath("logs")
  if not log_directory.exists():
    log_directory.mkdir()
  logger = logging.getLogger(logger_name)
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter(
      "%(asctime)s:%(levelname)s : %(name)s : %(message)s"
  )
  preprocessing_log_filepath = log_directory.joinpath(f"{logger_name}_logs.log")
  file_handler = logging.FileHandler(preprocessing_log_filepath)
  file_handler.setFormatter(formatter)
  if logger.hasHandlers():
    logger.handlers.clear()
  logger.addHandler(file_handler)
  return logger
