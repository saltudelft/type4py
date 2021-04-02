# Type4Py: Deep Similarity Learning-Based Type Inference for Python
This repository contains the implementation of Type4Py and instructions for re-producing the results of the paper.

- [Dataset](#dataset)
- [Installation Guide](#installation-guide)
- [Usage Guide](#usage-guide)
- [Citing Type4Py](#citing-type4py)

# Dataset
Type4Py dataset can be downloaded from [here](https://surfdrive.surf.nl/files/index.php/s/KobWgHFgXUgW4rA). It contains around 4,910 Python projects from GitHub, which were cloned in October 2019.

## Code de-duplication
Same as the paper, it is essential to de-duplicate the dataset for avoiding duplication bias when training and testing the model. Check out the `CD4Py` [tool](https://github.com/saltudelft/CD4Py) for code de-duplication.

# Installation Guide
## Requirements
- Linux-based OS
- Python 3.5 or newer
- An NVIDIA GPU with CUDA support

## Quick Install
```
git clone https://github.com/saltudelft/type4py.git && cd type4py
pip install .
```

# Usage Guide
Follow the below steps to train and evaluate the Type4Py model.
## 1. Extraction
```
$ type4py extract --c $DATA_PATH --o $OUTPUT_DIR --d $DUP_FILES --w $CORES
```
Description:
- `$DATA_PATH`: The path to the Python corpus or dataset.
- `$OUTPUT_DIR`: The path to store processed projects.
- `$DUP_FILES`: The path to the duplicate files. [Optional]
- `$CORES`: Number of CPU cores to use for processing projects.

## 2. Preprocessing
```
$ type4py preprocess --o $OUTPUT_DIR --l $LIMIT
```
Description:
- `$OUTPUT_DIR`: The path that was used in the first step to store processed projects.
- `$LIMIT`: The number of projects to be processed. [Optional]

## 3. Vectorizing
```
$ type4py vectorize --o $OUTPUT_DIR
```
Description:
- `$OUTPUT_DIR`: The path that was used in the first step to store processed projects.

## 4. Learning
```
$ type4py learn --o $OUTPUT_DIR --c --p $PARAM_FILE
```
Description:
- `$OUTPUT_DIR`: The path that was used in the first step to store processed projects.
- `--c`: Trains the model for the combined prediction task. Use `--a` and `--r` for argument and return type prediction tasks, respectively.
- `--p $PARAM_FILE`: The path to user-provided hyper-parameters for the model. See [this](https://github.com/saltudelft/type4py/blob/main/type4py/model_params.json) file as an example. [Optional]

## 5. Testing
```
$ type4py predict --o $OUTPUT_DIR --c
```

Description:
- `$OUTPUT_DIR`: The path that was used in the first step to store processed projects.
- `--c`: Tests the model for the combined prediction task. Use `--a` and `--r` for argument and return type prediction tasks, respectively. Note that this argument should be the same as the one that was used in the learning step.

## 6. Evaluating
```
$ type4py eval --o $OUTPUT_DIR --c --tp 10
```

Description:
- `$OUTPUT_DIR`: The path that was used in the first step to store processed projects.
- `--c`: Evaluates the model for the combined prediction task. Use `--a` and `--r` for argument and return type prediction tasks, respectively. Note that this argument should be the same as the one that was used in the learning step.
- `--tp 10`: Considers Top-10 predictions for evaluation. For this argument, You can choose a positive integer between 1 and 10. [Optional]

# Citing Type4Py

```
@article{mir2021type4py,
  title={Type4Py: Deep Similarity Learning-Based Type Inference for Python},
  author={Mir, Amir M and Latoskinas, Evaldas and Proksch, Sebastian and Gousios, Georgios},
  journal={arXiv preprint arXiv:2101.04470},
  year={2021}
}
```