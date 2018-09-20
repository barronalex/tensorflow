# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Model script to test TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class DilationTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Testing Conv2d with dilations in TF-TRT conversion."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [5, 8, 8, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtype=dtype, shape=input_dims, name=input_name)
      f1 = constant_op.constant(np.random.randn(1, 1, 2, 6), dtype=dtype)
      x1 = gen_nn_ops.conv2d(x, f1, [1, 1, 1, 1], 'SAME',
                             dilations=[1, 4, 4, 1])
      x1 = math_ops.sigmoid(x1)
      f2 = constant_op.constant(np.random.randn(1, 1, 6, 6), dtype=dtype)
      x2 = gen_nn_ops.conv2d(x1, f2, [1, 1, 1, 1], 'SAME',
                             dilations=[1, 2, 2, 1])
      x2 = math_ops.sigmoid(x2)
      f3 = constant_op.constant(np.random.randn(1, 1, 6, 6), dtype=dtype)
      x3 = gen_nn_ops.conv2d(x2, f3, [1, 1, 1, 1], 'SAME',
                             dilations=[1, 8, 2, 1])
      x3 = math_ops.sigmoid(x3)
      f4 = constant_op.constant(np.random.randn(1, 1, 6, 6), dtype=dtype)
      x4 = gen_nn_ops.conv2d(x3, f4, [1, 1, 1, 1], 'SAME',
                             dilations=[1, 2, 8, 1])
      out = array_ops.squeeze(x4, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(5, 8, 8, 6)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["my_trt_op_0"]

  def _ValidEngines(self):
    """Engines expected to build and run."""
    return ["my_trt_op_0"]


if __name__ == "__main__":
  test.main()
