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

from tensorflow.contrib.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ActivationTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Testing Activations in TF-TRT conversion."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [2, 3, 3, 1]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtype=dtype, shape=input_dims, name=input_name)
      x1 = math_ops.sigmoid(x)
      x2 = math_ops.tanh(x1)
      x3 = gen_nn_ops.relu(x2)
      x4 = math_ops.sigmoid(x3)
      x5 = math_ops.tanh(x4)
      x6 = gen_nn_ops.relu(x5, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(2, 3, 3, 1)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["my_trt_op_0"]

  def _ValidEngines(self):
    """Engines expected to build and run."""
    return ["my_trt_op_0"]


if __name__ == "__main__":
  test.main()
