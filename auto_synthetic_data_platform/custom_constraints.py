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

"""A custom constraints specification module of the Auto Synthetic Data Platform."""
import dataclasses
import pandas as pd
from typing import Callable

@dataclasses.dataclass()
class CustomConstraints:
    validity_func:Callable[[pd.DataFrame], pd.Series]
    custom_sampling_patience:int = 30
    custom_constraint_count:int = 500
