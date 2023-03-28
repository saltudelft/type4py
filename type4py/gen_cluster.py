import argparse
import os

from type4py.learn import load_model, TripletModel

from type4py.data_loaders import select_data, TripletDataset, load_test_data_per_model, load_training_data_per_model_sep
from type4py.deploy.infer import compute_types_score
from type4py.utils import load_model_params, setup_logs_file
from type4py import logger, MIN_DATA_POINTS, KNN_TREE_SIZE, data_loaders
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

class ModelNotfound(Exception):
    pass

class ModelNotExistsError(ModelNotfound):
    def __init__(self, model_name):
        super().__init__(f"Model {model_name} not found!")

class Type4Py(nn.Module):
    """
    Complete model
    """

    def __init__(self, input_size: int, hidden_size: int, aval_type_size: int,
                 num_layers: int, output_size: int, dropout_rate: float):
        super(Type4Py, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.aval_type_size = aval_type_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm_id = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                               bidirectional=True)
        self.lstm_tok = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                                bidirectional=True)
        self.linear = nn.Linear(self.hidden_size * 2 * 2 + self.aval_type_size, self.output_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_id, x_tok, x_type):
        # Using dropout on input sequences
        x_id = self.dropout(x_id)
        x_tok = self.dropout(x_tok)

        # Flattens LSTMs weights for data-parallelism in multi-GPUs config
        self.lstm_id.flatten_parameters()
        self.lstm_tok.flatten_parameters()

        x_id, _ = self.lstm_id(x_id)
        x_tok, _ = self.lstm_tok(x_tok)

        # Decode the hidden state of the last time step
        x_id = x_id[:, -1, :]
        x_tok = x_tok[:, -1, :]

        x = torch.cat((x_id, x_tok, x_type), 1)

        x = self.linear(x)
        return x


class TripletModel(nn.Module):
    """
    A model with Triplet loss for similarity learning
    """

    def __init__(self, model: nn.Module):
        super(TripletModel, self).__init__()
        self.model = model

    def forward(self, a, p, n):
        """
        A triplet consists of anchor, positive examples and negative examples
        """
        # return self.model(*(s.to(DEVICE) for s in a)), \
        #        self.model(*(s.to(DEVICE) for s in p)), \
        #        self.model(*(s.to(DEVICE) for s in n))

        return self.model(*(s for s in a)), \
               self.model(*(s for s in p)), \
               self.model(*(s for s in n))


def predict_type_embed(types_embed_array: np.array, types_embed_labels: np.array,
                       indexed_knn: AnnoyIndex, k: int) -> List[dict]:
    """
    Predict type of given type embedding vectors
    """

    pred_types_embed = []
    pred_types_score = []
    for i, embed_vec in enumerate(
            tqdm(types_embed_array, total=len(types_embed_array), desc="Finding KNNs & Prediction")):
        idx, dist = indexed_knn.get_nns_by_vector(embed_vec, k, include_distances=True)
        pred_idx_scores = compute_types_score(dist, idx, types_embed_labels)
        pred_types_embed.append([i for i, s in pred_idx_scores])
        pred_types_score.append(pred_idx_scores)

    return pred_types_embed, pred_types_score


def predict_type_embed_task(types_embed_array: np.array, types_embed_labels: np.array, type_space_labels: np.array,
                            pred_task_idx: tuple, indexed_knn: AnnoyIndex, k: int) -> List[dict]:
    def find_pred_task(i: int):
        if i < pred_task_idx[0]:
            return 'Parameter'
        elif i < pred_task_idx[1]:
            return 'Return'
        else:
            return 'Variable'

    pred_types: List[dict] = []
    # pred_types_embed = []
    # pred_types_score = []
    for i, embed_vec in enumerate(
            tqdm(types_embed_array, total=len(types_embed_array), desc="Finding KNNs & Prediction")):
        idx, dist = indexed_knn.get_nns_by_vector(embed_vec, k, include_distances=True)
        pred_idx_scores = compute_types_score(dist, idx, type_space_labels)

        pred_types.append({'original_type': types_embed_labels[i], 'predictions': pred_idx_scores,
                           'task': find_pred_task(i),
                           'is_parametric': bool(re.match(r'(.+)\[(.+)\]', types_embed_labels[i]))})

        # pred_types_embed.append([i for i, s in pred_idx_scores])
        # pred_types_score.append(pred_idx_scores)

    return pred_types


def build_type_clusters(model, output_path, train_data_loader: DataLoader, valid_data_loader: DataLoader, type_vocab: set,
                        exist_index: str, exist_emd: str):
    logger.info("Type Cluster building begin...")
    computed_embed_labels = []
    annoy_idx = AnnoyIndex(model.output_size, 'euclidean')
    loaded_idx = AnnoyIndex(model.output_size, 'euclidean')
    curr_idx = 0

    if exist_index is not None:
        loaded_idx.load(join(output_path, exist_index))
        curr_idx = loaded_idx.get_n_items()
        for i in range(loaded_idx.get_n_items()):
            item_vector = loaded_idx.get_item_vector(i)
            annoy_idx.add_item(i, item_vector)

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
                    annoy_idx.add_item(curr_idx, v)
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
                    annoy_idx.add_item(curr_idx, v)
                    computed_embed_labels.append(lables[i])
                    curr_idx += 1

    annoy_idx.build(KNN_TREE_SIZE)
    # annoy_idx.
    return annoy_idx, np.array(computed_embed_labels)  # np.hstack(computed_embed_labels)


def compute_type_embed_batch(model, data_loader: DataLoader, pca: PCA = None) -> Tuple[np.array, np.array]:
    """
    Compute type embeddings for the whole dataset
    """

    computed_embed_batches = []
    computed_embed_labels = []

    for batch_i, (a, p, n) in enumerate(tqdm(data_loader, total=len(data_loader), desc="Computing Type Clusters")):
        model.eval()
        with torch.no_grad():
            output_a = model(*(s.to(DEVICE) for s in a[0]))
            output_a = output_a.data.cpu().numpy()
            computed_embed_batches.append(pca.transform(output_a) if pca is not None else output_a)
            computed_embed_labels.append(a[1].data.cpu().numpy())

    return np.vstack(computed_embed_batches), np.hstack(computed_embed_labels)


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


def gen_cluster(output_path: str, data_loading_funcs: dict, datatype: str, type_vocab_limit: int = None,
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
