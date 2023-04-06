import argparse
import os

from type4py.learn import load_model, TripletModel, Type4Py
from type4py.predict import predict_type_embed, predict_type_embed_task

from type4py.data_loaders import select_data, TripletDataset, load_test_data_per_model, load_training_data_per_model_sep
from type4py.deploy.infer import compute_types_score
from type4py.utils import load_model_params, setup_logs_file
from type4py import logger, MIN_DATA_POINTS, KNN_TREE_SIZE, data_loaders
from type4py.exceptions import ModelNotExistsError
from libsa4py.utils import save_json
from typing import Tuple, List
from os.path import join
from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from annoy import AnnoyIndex
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import pickle
import re
import torch
import torch.nn as nn

logger.name = __name__
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_type_clusters(model, output_path, train_data_loader: DataLoader, valid_data_loader: DataLoader, type_vocab: set,
                        exist_index: str, exist_emd: str):
    logger.info("Type Cluster building begin...")
    computed_embed_labels = []
    current_annoy_idx = AnnoyIndex(model.output_size, 'euclidean')
    loaded_idx = AnnoyIndex(model.output_size, 'euclidean')
    curr_idx = 0

    if exist_index is not None:
        loaded_idx.load(join(output_path, exist_index))
        curr_idx = loaded_idx.get_n_items()
        for i in range(loaded_idx.get_n_items()):
            item_vector = loaded_idx.get_item_vector(i)
            current_annoy_idx.add_item(i, item_vector)

    if exist_emd is not None:
        embedd_labels = np.load(join(output_path, exist_emd)).tolist()
        computed_embed_labels.extend(embedd_labels)

    for _, (a, _, _) in enumerate(
            tqdm(train_data_loader, total=len(train_data_loader), desc="Computing Type Clusters - Train set")):
        model.eval()
        with torch.no_grad():
            output_a = model(*(s.to(DEVICE) for s in a[0]))
            lables = a[1].data.cpu().numpy()
            # computed_embed_labels.append(lables)
            for i, v in enumerate(output_a.data.cpu().numpy()):
                if lables[i] in type_vocab:
                    current_annoy_idx.add_item(curr_idx, v)
                    computed_embed_labels.append(lables[i])
                    curr_idx += 1

    for _, (a, _, _) in enumerate(
            tqdm(valid_data_loader, total=len(valid_data_loader), desc="Computing Type Clusters - Valid set")):
        model.eval()
        with torch.no_grad():
            output_a = model(*(s.to(DEVICE) for s in a[0]))
            lables = a[1].data.cpu().numpy()
            # computed_embed_labels.append(a[1].data.cpu().numpy())
            for i, v in enumerate(output_a.data.cpu().numpy()):
                if lables[i] in type_vocab:
                    current_annoy_idx.add_item(curr_idx, v)
                    computed_embed_labels.append(lables[i])
                    curr_idx += 1

    current_annoy_idx.build(KNN_TREE_SIZE)
    return current_annoy_idx, np.array(computed_embed_labels)

class DataTypeNotExistError(Exception):
    pass


def find_existing_index(data_loading_funcs, output_path):
    prefix = f"type4py_{data_loading_funcs['name']}_type_cluster"
    for filename in os.listdir(output_path):
        if filename.startswith(prefix):
            logger.info(f"find existing TypeCluster file: {filename}!")
            middle = filename[len(prefix):]
            trained = middle.split("_")
            return filename, trained
    return None, None


def find_existing_embedding(data_loading_funcs, output_path):
    prefix = f"type4py_{data_loading_funcs['name']}_true"
    suffix = ".npy"
    for filename in os.listdir(output_path):
        if filename.startswith(prefix) and filename.endswith(suffix):
            logger.info(f"find existing Embedding file: {filename}!")
            middle = filename[:-len(suffix)]
            # trained = middle.split("_")
            return filename, middle
    return None, None


def gen_type_cluster(output_path: str, data_loading_funcs: dict, datatype: str, type_vocab_limit: int = None,
                use_tc_reduced: bool = False):
    logger.info(f"Testing Type4Py model")
    logger.info(f"**********************************************************************")

    # Model's hyper parameters
    model_params = load_model_params()
    if os.path.exists(join(output_path, f"type4py_{data_loading_funcs['name']}_model_var_param_ret.pt")):
        model = torch.load(join(output_path, f"type4py_{data_loading_funcs['name']}_model_var_param_ret.pt"))
    else:
        raise ModelNotExistsError("type4py_{data_loading_funcs['name']}_model_var_param_ret.pt")

    le_all = pickle.load(open(join(output_path, "label_encoder_all.pkl"), 'rb'))
    type_vocab = pd.read_csv(join(output_path, '_most_frequent_all_types.csv')).head(
        type_vocab_limit if type_vocab_limit is not None else -1)
    type_vocab = set(le_all.transform(type_vocab['type'].values))
    logger.info(f"Loaded the pre-trained Type4Py {data_loading_funcs['name']} model")
    logger.info(f"Type4Py's trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    annoy_index: AnnoyIndex = None
    pca_transform: PCA = None
    embed_labels: np.array = None

    if not use_tc_reduced:

        # checking datatype
        if datatype not in {"var", "param", "ret"}:
            raise DataTypeNotExistError(f"datatype input {datatype} not in [ var, param, ret] list")

        # check existing AnnoyIndex and embedd_labels before generate new

        # checking and loading the existing Annoy_Index
        logger.info("Checking the existing AnnoyIndex...")
        cluster_file, processed_type = find_existing_index(data_loading_funcs, output_path)
        if cluster_file is None:
            logger.info("No existing AnnoyIndex found, started to initialising")

        # checking and loading the embedded labels
        logger.info("Checking the existing Embedding labels...")
        embedded_file, processed_type_em = find_existing_embedding(data_loading_funcs, output_path)
        if embedded_file is None:
            logger.info("No existing Embedding file found, started to initialising")

        # Loading dataset
        logger.info(f"Loading train and valid sets for datatype {datatype}")
        data_type_list = ["var", "param", "ret"]
        if datatype not in data_type_list:
            raise ValueError(f"{datatype} is not in the default data type list!")

        train_data_loader, valid_data_loader = load_training_data_per_model_sep(data_loading_funcs, output_path,
                                                                                datatype,
                                                                                model_params['batches'])
        logger.info(f"Train and Valid data loaded")

        # generate new anny_index and embed_labels
        annoy_index, embed_labels = build_type_clusters(model.model, output_path, train_data_loader, valid_data_loader, type_vocab,
                                                        cluster_file, embedded_file)
        logger.info("Created type clusters")

        # update and save the annoy_index and embed_labels
        if cluster_file is not None:
            os.remove(join(output_path,cluster_file))
            cluster_file = cluster_file + "_" + datatype
            annoy_index.save(join(output_path, cluster_file))
        else:
            annoy_index.save(join(output_path, f"type4py_{data_loading_funcs['name']}_type_cluster_{datatype}"))

        if embedded_file is not None:
            os.remove(join(output_path, embedded_file))
            embedded_file = processed_type_em + "_" + datatype
            np.save(join(output_path, embedded_file), embed_labels)
        else:
            np.save(join(output_path, f"type4py_{data_loading_funcs['name']}_true_{datatype}.npy"), embed_labels)

        logger.info("Saved type clusters")
