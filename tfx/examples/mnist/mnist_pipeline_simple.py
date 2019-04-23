# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Chicago taxi example using TFX."""

from __future__ import absolute_import, division, print_function

import datetime
import logging
import os

from tfx.components.example_gen.mnist_example_gen.component import \
    MnistExampleGen
from tfx.orchestration.airflow.airflow_runner import AirflowDAGRunner
from tfx.orchestration.pipeline import PipelineDecorator
from tfx.utils.dsl_utils import csv_input

_data_root = '/tmp/data/mnist/val/'

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines')
_metadata_db_root = os.path.join(_tfx_root, 'metadata')
_log_root = os.path.join(_tfx_root, 'logs')

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}

# Logging overrides
logger_overrides = {
    'log_root': _log_root,
    'log_level': logging.INFO
}


@PipelineDecorator(
    pipeline_name='chicago_taxi_simple',
    enable_cache=True,
    metadata_db_root=_metadata_db_root,
    additional_pipeline_args={'logger_args': logger_overrides},
    pipeline_root=_pipeline_root)
def _create_pipeline():
  """Implements the chicago taxi pipeline with TFX."""
  examples = csv_input(_data_root)

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = MnistExampleGen(input_base=examples)

  return [example_gen]


pipeline = AirflowDAGRunner(_airflow_config).run(_create_pipeline())
