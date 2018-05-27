#!/usr/bin/env bash
# : ${MODEL_NAME:=model}
# : ${PORT:=9000}
# if [ -z $SERVING_MODEL ]; then
#   export SERVING_MODEL=$MODEL_URL
# fi
set -xe

# Fetch models from storage
# stored sync $SERVING_MODEL /model/1/

# Start the model server
serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
  --port=9000 \
  --model_name=inception \
  --model_base_path=/tmp/inception-export