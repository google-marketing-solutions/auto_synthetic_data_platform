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
#!/usr/bin/env python

"""The setup script."""

import pathlib
from setuptools import find_packages
from setuptools import setup


with pathlib.Path('requirements.txt').open() as requirements_path:
  requirements = requirements_path.read().splitlines()

setup(
    author='Google EMEA gPS Data Science Team',
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    description=(
        "Google EMEA gPS Data Science Team's solution to create privacy-safe"
        ' synthetic data out of real data. The solution is a wrapper around the'
        ' synthcity package (https://github.com/vanderschaarlab/synthcity)'
        ' simplifying the process of model tuning.'
    ),
    install_requires=requirements,
    license='Apache Software License 2.0',
    packages=find_packages(
        include=[
            'auto_synthetic_data_platform',
            'auto_synthetic_data_platform.*',
        ]
    ),
    version='0.1.0',
)
