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

"""The setup script for Auto Synthetic Data Platform."""

from typing import Final
import os
import setuptools


_CURRENT_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))


def _get_readme():
    try:
        readme = open(os.path.join(_CURRENT_DIR, "README.md"), encoding="utf-8").read()
    except OSError:
        readme = ""
    return readme


def _get_version():
    with open(os.path.join(_CURRENT_DIR, "auto_synthetic_data_platform", "__init__.py")) as fp:
        for line in fp:
            if line.startswith("__version__") and "=" in line:
                version = line[line.find("=") + 1 :].strip(" '\"\n")
                if version:
                    return version
        raise ValueError("`__version__` not defined in `auto_synthetic_data_platform/__init__.py`")


def _parse_requirements(path):
    with open(os.path.join(_CURRENT_DIR, path)) as f:
        return [
            line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))
        ]


_VERSION: Final[str] = _get_version()
_README: Final[str] = _get_readme()
_INSTALL_REQUIREMENTS: Final[str] = _parse_requirements(
    os.path.join(_CURRENT_DIR, "requirements.txt")
)


setuptools.setup(
    name="auto-synthetic-data-platform",
    version=_VERSION,
    python_requires=">=3.10",
    description=(
        "Google EMEA gTech Ads Data Science Team's solution to create privacy-safe"
        " synthetic data out of real data. The solution is a wrapper around the"
        " synthcity package (https://github.com/vanderschaarlab/synthcity)"
        " simplifying the process of model tuning."
    ),
    long_description=_README,
    long_description_content_type="text/markdown",
    author="Google gTech Ads EMEA Privacy Data Science Team",
    license="Apache Software License 2.0",
    packages=setuptools.find_packages(),
    install_requires=_INSTALL_REQUIREMENTS,
    url="https://github.com/google-marketing-solutions/auto_synthetic_data_platform",
    keywords="1pd privacy ai ml marketing",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
