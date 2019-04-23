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
"""Tests for tfx.components.example_gen.csv_example_gen.executor."""

from __future__ import absolute_import, division, print_function

import io
import os

import apache_beam as beam
import numpy as np
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions

from tfx.components.example_gen.mnist_example_gen.executor import (
    FEATURE_DESCRIPTION, TFRECORD_OUTFILE, _ImageToExample)
from tfx.examples.mnist.mnist_utils import maybe_download
from tfx.utils import types


def write_tfrecords():
  """
    Main write function
    """
  maybe_download()
  input_dict = {'input-base': '/tmp/data/mnist/val/'}
  with beam.Pipeline(options=PipelineOptions()) as p:
    tf_example = p | "InputSourceToExample" >> _ImageToExample(input_dict)

    serialize = (tf_example | 'SerializeDeterministically' >>
                 beam.Map(lambda x: x.SerializeToString(deterministic=True)))

    (serialize
     | beam.io.WriteToTFRecord(TFRECORD_OUTFILE, file_name_suffix='.gz'))


def get_raw_dataset(filename):
  filenames = [filename]
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def get_record(dataset):
  return next(iter(dataset.take(1)))


def _parse_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.parse_single_example(example_proto, FEATURE_DESCRIPTION)


def convert_parsed_record_to_ndarray(parsed_record):
  x = parsed_record['image_raw']
  x_np = x.numpy()
  bytestream = io.BytesIO(x_np)
  rows = 28
  cols = 28
  num_images = 1
  buf = bytestream.read(rows * cols * num_images)
  data = np.frombuffer(buf, dtype=np.uint8)
  shape = (rows, cols, num_images)
  data = data.reshape(*shape)
  assert isinstance(data, np.ndarray), type(data)
  assert data.shape == shape
  return data


def read_tfrecord():
  """
    Main read function

    Reads a single image TFRecord and returns it as a np.ndarray
    """
  tfrecord_infile = '{}-00000-of-00001.gz'.format(TFRECORD_OUTFILE)

  raw_dataset = get_raw_dataset(tfrecord_infile)

  parsed_dataset = raw_dataset.map(_parse_function)

  parsed_record = get_record(parsed_dataset)

  return convert_parsed_record_to_ndarray(parsed_record)


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    self.outfile = '{}-00000-of-00001.gz'.format(TFRECORD_OUTFILE)

    self._maybe_remove_file()

  def tearDown(self):
    self._maybe_remove_file()

  def _maybe_remove_file(self):
    if os.path.isfile(self.outfile):
      os.remove(self.outfile)

  def test_image_to_example_gen(self):
    self.assertFalse(os.path.isfile(self.outfile))

    # invokes _ImageToExample
    write_tfrecords()

    self.assertTrue(os.path.isfile(self.outfile))

    # read data back and check
    sample = read_tfrecord()

    self.assertIsInstance(sample, np.ndarray)
    self.assertEqual(sample.shape, (28, 28, 1))


if __name__ == '__main__':
  tf.test.main()
