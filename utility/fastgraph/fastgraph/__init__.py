"""
  Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
  
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import os

from .dataset_loader import DatasetLoader


def dataset(name, root_path, force_load64=False):
    assert(name in ['papers100M', 'com-friendster',
           'reddit', 'products', 'twitter', 'uk-2006-05'])
    dataset_path = os.path.join(root_path, name)
    dataset_loader = DatasetLoader(dataset_path, force_load64)
    return dataset_loader


__all__ = ['dataset']
