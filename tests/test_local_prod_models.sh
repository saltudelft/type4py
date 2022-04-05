#!/bin/bash
set -e
# This script tests both local and production pre-trained models in an end-to-end fashion using Docker images (an integration test).
# $1: Path to the local model
# $2: Path to the production model

cd ../
# Local model
echo "#####################################Testing Local Model##########################################"
if [ -n "$1" ]; then
    tar -xzvf "$1" ./
    docker build --no-cache -t type4py -f Dockerfile .
    local_container_id=$(docker run -d -p 5001:5010 type4py)
    cd ./type4py/server/tests/ && pytest test_local_server.py && cd -
    docker stop "$local_container_id"
    rm -rf t4py_model_files
else
 echo "Error: Path to the local model files not given!"
 exit 1
fi

# Production model - CPU
echo "#####################################Testing Production Model - CPU##########################################"
if [ -n "$2" ]; then
    docker build --no-cache -t type4py.prod.cpu -f Dockerfile .
    prod_container_id=$(docker run -d -p 5001:5010 -v "$2":"/t4py_model_files/" --env-file ./t4py_env.list type4py.prod.cpu)
    cd ./type4py/server/tests/ && pytest test_server.py --env "local" && cd -
    docker stop "$prod_container_id"
else
 echo "Error: Path to the production model files not given!"
 exit 1
fi

# Production model - GPU
echo "#####################################Testing Production Model - GPU##########################################"
docker build --no-cache -t type4py.prod.gpu -f Dockerfile.cuda .
prod_container_id=$(docker run -d -p 5001:5010 --gpus "device=1" -v "$2":"/t4py_model_files/" --env-file ./t4py_env.list  type4py.prod.gpu)
cd ./type4py/server/tests/ && pytest test_server.py --env "local" && cd -
docker stop "$prod_container_id"

echo -e "\033[0;32mSucceeded! \033[0m"