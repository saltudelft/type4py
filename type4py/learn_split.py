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
from os.path import join, exists
from time import time
from annoy import AnnoyIndex
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import pickle

logger.name = __name__
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_pickle_file(data_loading_funcs, output_path):
    prefix = f"{data_loading_funcs['name']}_common_types"
    suffix = "pkl"
    for filename in os.listdir(output_path):
        if filename.startswith(prefix) and filename.endswith(suffix):
            logger.info(f"find existing common types file: {filename}!")
            middle = filename[len(prefix):-len(suffix)]
            trained = middle.split("_")
            return filename, trained
    return None, None


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

    common_typefile, common_datatype = check_pickle_file(data_loading_funcs, output_path)
    if common_datatype == None:
        common_typefile = f"{data_loading_funcs['name']}_common_types.pkl"

    else:
        logger.info(f"Load existing {common_typefile} file !")
        with open(join(output_path, common_typefile), 'rb') as f1:
            count_types_pre = pickle.load(f1)
        count_types.update(count_types_pre)


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
    # saving common types
    logger.info("Saving common types...")
    with open(join(output_path, f"{common_typefile[:-4]}_{dataset_type}.pkl"), 'wb') as f:
        pickle.dump(common_types, f)
    # remove old common types
    if common_datatype is not None:
        os.remove(join(output_path, common_typefile))



    # get the trained_model name and trained_types
    trained_model_name, trained_types = find_existing_model(data_loading_funcs, output_path)

    if trained_types == None:
        trained_model_name = f"type4py_{data_loading_funcs['name']}_model.pt"
        logger.info(f"No trained model found, starting to initialize the model {trained_model_name}...")
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
    torch.save(model.module if torch.cuda.device_count() > 1 else model,
               join(output_path, f"{trained_model_name[:-3]}_{dataset_type}.pt"))
    # remove old model
    if exists(join(output_path, trained_model_name)):
        os.remove(join(output_path, trained_model_name))
