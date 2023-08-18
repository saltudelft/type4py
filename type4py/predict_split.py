import os

from type4py.data_loaders import select_data, TripletDataset, load_training_data_per_model, load_test_data_per_model
from type4py.deploy.infer import compute_types_score
from type4py.learn import load_model, TripletModel, Type4Py
from type4py.exceptions import ModelNotExistsError
from type4py.utils import load_model_params
from type4py import logger, MIN_DATA_POINTS, KNN_TREE_SIZE
from type4py.predict import compute_type_embed_batch, predict_type_embed_task
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
import torch

logger.name = __name__
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_split(output_path: str, data_loading_funcs: dict):

    logger.info(f"Testing Type4Py model")
    logger.info(f"**********************************************************************")

    # Model's hyper parameters
    model_params = load_model_params()
    if os.path.exists(join(output_path, f"type4py_{data_loading_funcs['name']}_model_var_param_ret.pt")):
        model = torch.load(join(output_path, f"type4py_{data_loading_funcs['name']}_model_var_param_ret.pt"))
    else:
        raise ModelNotExistsError("type4py_{data_loading_funcs['name']}_model_var_param_ret.pt")
    le_all = pickle.load(open(join(output_path, "label_encoder_all.pkl"), 'rb'))
    logger.info(f"Loaded the pre-trained Type4Py {data_loading_funcs['name']} model")
    logger.info(f"Type4Py's trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    annoy_index: AnnoyIndex = None
    pca_transform: PCA = None
    embed_labels: np.array = None


    logger.info("Loading the reduced type clusters")

    pca_transform = pickle.load(open(join(output_path, "type_clusters_pca.pkl"), 'rb'))
    embed_labels = np.load(join(output_path, f"type4py_{data_loading_funcs['name']}_true_var_param_ret.npy"))
    embed_labels = np.array(embed_labels, dtype=int)
    annoy_index = AnnoyIndex(pca_transform.n_components_, 'euclidean')
    annoy_index.load(join(output_path, "type4py_complete_type_cluster_reduced"))

    logger.info("Loading test set")
    test_data_loader, t_idx = load_test_data_per_model(data_loading_funcs, output_path, model_params['batches_test'])
    logger.info("Mapping test samples to type clusters")

    test_type_embed, embed_test_labels = compute_type_embed_batch(model.model, test_data_loader, pca_transform)

    # Perform KNN search and predict
    logger.info("Performing KNN search")

    train_valid_labels = le_all.inverse_transform(embed_labels)
    embed_test_labels = le_all.inverse_transform(embed_test_labels)
    pred_types = predict_type_embed_task(test_type_embed, embed_test_labels,
                                         train_valid_labels,
                                         t_idx, annoy_index, model_params['k'])

    save_json(join(output_path, f"type4py_{data_loading_funcs['name']}_test_predictions.json"), pred_types)
    logger.info("Saved the Type4Py model's predictions on the disk")
