import os

from type4py.data_loaders import select_data, TripletDataset, load_training_data_per_model, \
    load_training_data_per_model_sep
from type4py.vectorize import AVAILABLE_TYPES_NUMBER, W2V_VEC_LENGTH
from type4py.learn import load_model, TripletModel, Type4Py, create_knn_index, train_loop_dsl
from type4py.eval import eval_type_embed
from type4py.utils import load_model_params
from type4py import logger, MIN_DATA_POINTS, KNN_TREE_SIZE
from type4py.exceptions import ModelTrainedError
from torch.utils.data import DataLoader
from typing import Tuple
from collections import Counter
from multiprocessing import cpu_count
from os.path import join
from time import time
from annoy import AnnoyIndex
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import pickle

logger.name = __name__
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_pickle_file(type, data_loading_funcs, output_path):
    var_exist = False
    param_exist = False
    ret_exist = False
    if os.path.exists(join(output_path, f"{data_loading_funcs['name']}_common_types_var.pkl")) and type != "var":
        var_exist = True
        logger.info(f"find existing {data_loading_funcs['name']}_common_types_var.pkl file !")
    if os.path.exists(join(output_path, f"{data_loading_funcs['name']}_common_types_param.pkl")) and type != "param":
        param_exist = True
        logger.info(f"find existing {data_loading_funcs['name']}_common_types_param.pkl file !")
    if os.path.exists(join(output_path, f"{data_loading_funcs['name']}_common_types_ret.pkl")) and type != "ret":
        ret_exist = True
        logger.info(f"find existing {data_loading_funcs['name']}_common_types_ret.pkl file !")
    return var_exist, param_exist, ret_exist


# find existing trained model, return trained_types
def find_existing_model(data_loading_funcs, output_path):
    prefix = f"type4py_{data_loading_funcs['name']}_model"
    suffix = ".pt"
    for filename in os.listdir(output_path):
        if filename.startswith(prefix) and filename.endswith(suffix):
            logger.info(f"find existing model file: {filename}!")
            middle = filename[len(prefix):-len(suffix)]
            trained = middle.split("_")
            return filename, trained
    return None, None


def train_split(output_path: str, data_loading_funcs: dict, dataset_type: str, model_params_path=None,
                validation: bool = False):
    logger.info(f"Training Type4Py model")
    logger.info(f"***********************************************************************")

    # Model's hyper parameters
    model_params = load_model_params(model_params_path)

    # data loading process based on datatype
    data_type_list = ["var", "param", "ret"]
    if dataset_type not in data_type_list:
        raise ValueError(f"{dataset_type} is not in the default data type list!")

    train_data_loader, valid_data_loader = load_training_data_per_model_sep(data_loading_funcs, output_path,
                                                                            dataset_type,
                                                                            model_params['batches'],
                                                                            load_valid_data=validation,
                                                                            no_workers=cpu_count() // 2)

    # Loading label encoder and check existing count_types file
    le_all = pickle.load(open(join(output_path, "label_encoder_all.pkl"), 'rb'))
    count_types = Counter(train_data_loader.dataset.labels.data.numpy())

    var_exists, param_exits, ret_exists = check_pickle_file(dataset_type, data_loading_funcs, output_path)

    if os.path.exists(join(output_path, f"{data_loading_funcs['name']}_common_types_{dataset_type}.pkl")):
        logger.warn(f"{data_loading_funcs['name']}_common_types_{dataset_type}.pkl file exists!")

    with open(join(output_path, f"{data_loading_funcs['name']}_common_types_{dataset_type}.pkl"), 'wb') as f:
        pickle.dump(count_types, f)

    type_filename = dataset_type

    # if find existing types in "var" dataset, load them for updating for final common types
    if var_exists and dataset_type != "var":
        with open(join(output_path, f"{data_loading_funcs['name']}_common_types_var.pkl"), 'rb') as f1:
            count_types_var = pickle.load(f1)
        count_types.update(count_types_var)
        # also add suffix to filename
        type_filename = type_filename + "_var"

    # if find existing types in "param" dataset, load them for updating for final common types
    if param_exits and dataset_type != "param":
        with open(join(output_path, f"{data_loading_funcs['name']}_common_types_param.pkl"), 'rb') as f2:
            count_types_param = pickle.load(f2)
        count_types.update(count_types_param)
        type_filename = type_filename + "_param"

    # if find existing types in "ret" dataset, load them for updating for final common types
    if ret_exists and dataset_type != "ret":
        with open(join(output_path, f"{data_loading_funcs['name']}_common_types_ret.pkl"), 'rb') as f3:
            count_types_ret = pickle.load(f3)
        count_types.update(count_types_ret)
        type_filename = type_filename + "_ret"

    common_types = [t.item() for t in train_data_loader.dataset.labels if count_types[t.item()] >= 100]
    ubiquitous_types = set(le_all.transform(['str', 'int', 'list', 'bool', 'float']))
    common_types = set(common_types) - ubiquitous_types

    logger.info("Percentage of ubiquitous types: %.2f%%" % (len([t.item() for t in \
                                                                 train_data_loader.dataset.labels if
                                                                 t.item() in ubiquitous_types]) /
                                                            train_data_loader.dataset.labels.shape[0] * 100.0))
    logger.info("Percentage of common types: %.2f%%" % (len([t.item() for t in \
                                                             train_data_loader.dataset.labels if
                                                             t.item() in common_types]) /
                                                        train_data_loader.dataset.labels.shape[0] * 100.0))

    with open(join(output_path, f"{data_loading_funcs['name']}_common_types_{type_filename}.pkl"), 'wb') as f:
        pickle.dump(common_types, f)

    # get the trained_model name and trained_types
    trained_model_name, trained_types = find_existing_model(data_loading_funcs, output_path)

    if trained_types == None:
        logger.info("No trained model found, starting to intialize the model...")
        # Loading the model
        model = load_model(data_loading_funcs['name'], model_params)
        logger.info(f"Intializing the {model.__class__.__name__} model")
        model = TripletModel(model).to(DEVICE)
    else:
        if dataset_type in trained_types:
            raise ModelTrainedError
        else:
            logger.info(f"Loading saved model {trained_model_name}...")
            model = torch.load(join(output_path, trained_model_name))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    logger.info(f"Model training on {DEVICE}")

    criterion = torch.nn.TripletMarginLoss(margin=model_params['margin'])
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['lr'])

    train_t = time()
    train_loop_dsl(model, criterion, optimizer, train_data_loader,
                   valid_data_loader if validation else None, model_params['lr'],
                   model_params['epochs'], ubiquitous_types, common_types, None)
    logger.info("Training finished in %.2f min" % ((time() - train_t) / 60))

    # Saving the model
    logger.info("Saved the trained Type4Py model for %s prediction on the disk" % data_loading_funcs['name'])
    if trained_model_name == None:
        trained_model_name == f"type4py_{data_loading_funcs['name']}_model.pt"
    torch.save(model.module if torch.cuda.device_count() > 1 else model,
               join(output_path, f"{trained_model_name[:-3]}_{dataset_type}.pt"))
