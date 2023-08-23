# Type4Py: Deep Similarity Learning-Based Type Inference for Python
![GH Workflow](https://github.com/saltudelft/type4py/actions/workflows/.github/workflows/type4py_server_test.yaml/badge.svg)
![GH Workflow](https://github.com/saltudelft/type4py/actions/workflows/.github/workflows/publish_type4py_docker_img.yaml/badge.svg)

This repository contains the implementation of Type4Py and instructions for re-producing the results of the paper.

- [Dataset](#dataset)
- [Installation Guide](#installation-guide)
- [Usage Guide](#usage-guide)
- [Converting Type4Py to ONNX](#converting-type4py-to-onnx)
- [VSCode Extension](#vscode-extension)
- [Using Local Pre-trained Model](#using-local-pre-trained-model)
- [Type4Py Server](#type4py-server)
- [Citing Type4Py](#citing-type4py)

# Dataset
For Type4Py, we use the **ManyTypes4Py** dataset. You can download the latest version of the dataset [here](https://doi.org/10.5281/zenodo.4044635).
Also, note that the dataset is already de-duplicated.

## Code De-deduplication
If you want to use your own dataset, 
it is essential to de-duplicate the dataset by using a tool like [CD4Py](https://github.com/saltudelft/CD4Py).

# Installation Guide
## Requirements
Here are the recommended system requirements for training Type4Py on the MT4Py dataset:
- Linux-based OS (Ubuntu 18.04 or newer)
- Python 3.6 or newer
- A high-end NVIDIA GPU (w/ at least 8GB of VRAM)
- A CPU with 16 threads or higher (w/ at least 64GB of RAM)

## Quick Install
```
git clone https://github.com/saltudelft/type4py.git && cd type4py
pip install .
```

# Usage Guide
Follow the below steps to train and evaluate the Type4Py model.
## 1. Extraction
**NOTE:** Skip this step if you're using the ManyTypes4Py dataset.
```
$ type4py extract --c $DATA_PATH --o $OUTPUT_DIR --d $DUP_FILES --w $CORES
```
Description:
- `$DATA_PATH`: The path to the Python corpus or dataset.
- `$OUTPUT_DIR`: The path to store processed projects.
- `$DUP_FILES`: The path to the duplicate files, i.e., the `*.jsonl.gz` file produced by CD4Py. [Optional]
- `$CORES`: Number of CPU cores to use for processing projects.

## 2. Preprocessing
```
$ type4py preprocess --o $OUTPUT_DIR --l $LIMIT
```
Description:
- `$OUTPUT_DIR`: The path that was used in the first step to store processed projects. For the MT4Py dataset, use the directory in which the dataset is extracted.
- `$LIMIT`: The number of projects to be processed. [Optional]

## 3. Vectorizing
```
$ type4py vectorize --o $OUTPUT_DIR
```
Description:
- `$OUTPUT_DIR`: The path that was used in the previous step to store processed projects.

[//]: # (## 4. Learning)

[//]: # (```)

[//]: # ($ type4py learn --o $OUTPUT_DIR --c --p $PARAM_FILE)

[//]: # (```)

[//]: # (Description:)

[//]: # (- `$OUTPUT_DIR`: The path that was used in the previous step to store processed projects.)

[//]: # (- `--c`: Trains the complete model. Use `type4py learn -h` to see other configurations.)

[//]: # ()
[//]: # (- `--p $PARAM_FILE`: The path to user-provided hyper-parameters for the model. See [this]&#40;https://github.com/saltudelft/type4py/blob/main/type4py/model_params.json&#41; file as an example. [Optional])

## 4*. Learning separately
```
$ type4py learns --o $OUTPUT_DIR --dt $DATA_TYPE --c --p $PARAM_FILE 
```
- `$OUTPUT_DIR`: The path that was used in the previous step to store processed projects.
- `$DATA_TYPE`: Sequential Learing, either `var`, or `param` or `ret`
- `--c`: Trains the complete model. Use `type4py learn -h` to see other configurations.

- `--p $PARAM_FILE`: The path to user-provided hyper-parameters for the model. See [this](https://github.com/saltudelft/type4py/blob/main/type4py/model_params.json) file as an example. [Optional]

## 5**. Gernerating Type Cluster
```
$ type4py gen_type_clu --o $OUTPUT_DIR --dt $DATA_TYPE 
```
- `$OUTPUT_DIR`: The path that was used in the previous step to store processed projects.
- `$DATA_TYPE`: Sequential Learing, either `var`, or `param` or `ret`

## 6. Reducing Type Cluster
To reduce the dimension of the created type clusters in step 5, run the following command:
> Note: The reduced version of type clusters causes a slight performance loss in type prediction.
```
$ type4py reduce --o $OUTPUT_DIR --d $DIMENSION
```

Description:
- `$OUTPUT_DIR`: The path that was used in the first step to store processed projects.
- `$DIMENSION`: Reduces the dimension of type clusters to the specified value [Default: 256]

## 7*. Project-base inference
```python
$ type4py infer_project --m results --p raw_projects --o results --a t4py
```
- `$--m`: The path that saved the model
- `$--p`:The path that saved the raw projects, for project-base inference
- `$--o`:The path that output the inference results
- `$--a`:The approach you want, including t4py, t4pyre, t4pyright
```python
$ type4py infer_project --m results --p raw_projects --o results --a t4pyre
```

## 7. Testing
```
$ type4py predicts --o $OUTPUT_DIR
```

Description:
- `$OUTPUT_DIR`: The path that was used in the first step to store processed projects.

[//]: # (- `--c`: Predicts using the complete model. Use `type4py predict -h` to see other configurations.)

## 8. Evaluating
```
$ type4py eval --o $OUTPUT_DIR --t c --tp 10
```

Description:
- `$OUTPUT_DIR`: The path that was used in the first step to store processed projects.
- `--t`: Evaluates the model considering different prediction tasks. E.g., `--t c` considers all predictions tasks,
  i.e., parameters, return, and variables. [Default: c]
- `--tp 10`: Considers Top-10 predictions for evaluation. For this argument, You can choose a positive integer between 1 and 10. [Default: 10]

Use `type4py eval -h` to see other options.


# Converting Type4Py to ONNX
To convert the pre-trained Type4Py model to the [ONNX](https://onnxruntime.ai/) format, use the following command:
```
$ type4py to_onnx --o $OUTPUT_DIR
```
Description:
- `$OUTPUT_DIR`: The path that was used in the [usage](#usage-guide) section to store processed projects and the model.

# VSCode Extension
[![vsm-version](https://img.shields.io/visual-studio-marketplace/v/saltud.type4py?style=flat&label=VS%20Marketplace&logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=saltud.type4py)

Type4Py can be used in VSCode, which provides ML-based type auto-completion for Python files. The Type4Py's VSCode extension can be installed from the VS Marketplace [here](https://marketplace.visualstudio.com/items?itemName=saltud.type4py).

# Using Local Pre-trained Model
Type4Py's pre-trained model can be queried locally by using provided Docker images. See [here](https://github.com/saltudelft/type4py/wiki/Type4Py's-Local-Model) for usage info.

# Type4Py Server
![GH Workflow](https://github.com/saltudelft/type4py/actions/workflows/.github/workflows/type4py_server_test.yaml/badge.svg)

The Type4Py server is deployed on our server, which exposes a public API and powers the VSCode extension.
However, if you would like to deploy the Type4Py server on your own machine, you can adapt the server code [here](https://github.com/saltudelft/type4py/tree/server/type4py/server).
Also, please feel free to reach out to us for deployment, using the pre-trained Type4Py model and how to train your own model by creating an [issue](https://github.com/saltudelft/type4py/issues).

# Citing Type4Py

```
@inproceedings{mir2022type4py,
  title={Type4Py: practical deep similarity learning-based type inference for python},
  author={Mir, Amir M and Lato{\v{s}}kinas, Evaldas and Proksch, Sebastian and Gousios, Georgios},
  booktitle={Proceedings of the 44th International Conference on Software Engineering},
  pages={2241--2252},
  year={2022}
}
```