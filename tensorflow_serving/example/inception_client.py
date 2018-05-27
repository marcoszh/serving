# Copyright 2016 Google Inc. All Rights Reserved.
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

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import sys
import threading
import logging
import time
from pprint import pprint

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_integer('concurrency', 10,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 1000, 'Number of test images')
tf.app.flags.DEFINE_integer('batch_size', 10, 'Number of test images')
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('time_out', '', 'timeout')
FLAGS = tf.app.flags.FLAGS

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.INFO, format=LOG_FORMAT)

class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._error = 0
    self._done = 0
    self._active = 0
    self._condition = threading.Condition()

  def inc_error(self):
    with self._condition:
      self._error += 1

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_error_rate(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._error / float(self._num_tests)

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1

def _create_rpc_callback(result_counter):
  """Creates RPC callback function.

  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.

    Calculates the statistics for the prediction result.

    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print(exception)
    else:
      # total_time = int((time.time() - start_time) * 1000)
      sys.stdout.write('.')
      sys.stdout.flush()
      #pprint(vars(result_future))
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback


def do_inference(hostport, concurrency, num_tests, batch_size):
  """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  result_counter = _ResultCounter(num_tests, concurrency)
  logging.info("Start testing with " + str(num_tests) + " " + str(concurrency))
  with open(FLAGS.image, 'rb') as f:
    data = f.read()
    data_array = []
    for ic in range(batch_size):
      image_data.append(data_array)
    for _ in range(num_tests):
      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'inception'
      request.model_spec.signature_name = 'predict_images'
      request.inputs['images'].CopyFrom(
          tf.contrib.util.make_tensor_proto(data_array, shape=[len(data_array)]))
      result_counter.throttle()
      start_time = time.time()
      result_future = stub.Predict.future(request, float(FLAGS.time_out))  # 5 seconds
      result_future.add_done_callback(
          _create_rpc_callback(result_counter))
  return result_counter.get_error_rate()


def main(_):
  start_time = time.time()
  error_rate = do_inference(FLAGS.server,
                            FLAGS.concurrency, FLAGS.num_tests, FALGS.batch_size)
  total_time = int((time.time() - start_time) * 1000)
  logging.info("test done")
  print('\nInference error rate: %s%%' % (error_rate * 100))
  print('\nTime elapsed:%s' % total_time)
  # host, port = FLAGS.server.split(':')
  # channel = implementations.insecure_channel(host, int(port))
  # stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # Send request
  # with open(FLAGS.image, 'rb') as f:
    # See prediction_service.proto for gRPC request/response details.
    # data = f.read()
    # request = predict_pb2.PredictRequest()
    # request.model_spec.name = 'inception'
    # request.model_spec.signature_name = 'predict_images'
    # request.inputs['images'].CopyFrom(
    #     tf.contrib.util.make_tensor_proto(data, shape=[1]))
    # result = stub.Predict(request, 10.0)  # 10 secs timeout
    # print(result)


if __name__ == '__main__':
  tf.app.run()
