from type4py.data_loaders import select_data, TripletDataset, load_training_data_per_model, load_test_data_per_model
from type4py.learn import load_model_params, TripletModel, create_knn_index
from type4py import logger, MIN_DATA_POINTS, KNN_TREE_SIZE
from libsa4py.utils import save_json
from typing import Tuple, List
from collections import defaultdict
from os.path import join
from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from annoy import AnnoyIndex
import numpy as np
import pickle
import re
import torch

logger.name = __name__
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_types_score(types_dist: list, types_idx: list, types_embed_labels: np.array):
        types_dist = 1 / (np.array(types_dist) + 1e-10) ** 2
        types_dist /= np.sum(types_dist)
        types_score = defaultdict(int)
        for n, d in zip(types_idx, types_dist):
            types_score[types_embed_labels[n]] += d
        
        return sorted({t: s for t, s in types_score.items()}.items(), key=lambda kv: kv[1],
                      reverse=True)

def predict_type_embed(types_embed_array: np.array, types_embed_labels: np.array, 
                       indexed_knn: AnnoyIndex, k: int) -> List[dict]:
    """
    Predict type of given type embedding vectors
    """

    pred_types_embed = []
    pred_types_score = []
    for i, embed_vec in enumerate(tqdm(types_embed_array, total=len(types_embed_array), desc="Finding KNNs & Prediction")):
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
    for i, embed_vec in enumerate(tqdm(types_embed_array, total=len(types_embed_array), desc="Finding KNNs & Prediction")):
        idx, dist = indexed_knn.get_nns_by_vector(embed_vec, k, include_distances=True)
        pred_idx_scores = compute_types_score(dist, idx, type_space_labels)

        pred_types.append({'original_type': types_embed_labels[i], 'predictions': pred_idx_scores,
                           'task': find_pred_task(i), 'is_parametric': bool(re.match(r'(.+)\[(.+)\]', types_embed_labels[i]))})

        # pred_types_embed.append([i for i, s in pred_idx_scores])
        # pred_types_score.append(pred_idx_scores)
    
    return pred_types


def build_type_clusters(model, train_data_loader: DataLoader, valid_data_loader: DataLoader):

    computed_embed_labels = []
    annoy_idx = AnnoyIndex(model.output_size, 'euclidean')
    curr_idx = 0

    for _, (a, _, _) in enumerate(tqdm(train_data_loader, total=len(train_data_loader), desc="Computing Type Clusters - Train set")):
        model.eval()
        with torch.no_grad():
            output_a = model(*(s.to(DEVICE) for s in a[0]))
            computed_embed_labels.append(a[1].data.cpu().numpy())
            for v in output_a.data.cpu().numpy():
                annoy_idx.add_item(curr_idx, v)
                curr_idx += 1

    for _, (a, _, _) in enumerate(tqdm(valid_data_loader, total=len(valid_data_loader), desc="Computing Type Clusters - Valid set")):
        model.eval()
        with torch.no_grad():
            output_a = model(*(s.to(DEVICE) for s in a[0]))
            computed_embed_labels.append(a[1].data.cpu().numpy())
            for v in output_a.data.cpu().numpy():
                annoy_idx.add_item(curr_idx, v)
                curr_idx += 1

    annoy_idx.build(KNN_TREE_SIZE)
    return annoy_idx, np.hstack(computed_embed_labels)

def compute_type_embed_batch(model, data_loader: DataLoader) -> Tuple[np.array, np.array]:
    """
    Compute type embeddings for the whole dataset
    """

    computed_embed_batches = []
    computed_embed_labels = []

    for batch_i, (a, p, n) in enumerate(tqdm(data_loader, total=len(data_loader), desc="Computing Type Clusters")):
        model.eval()
        with torch.no_grad():
            output_a = model(*(s.to(DEVICE) for s in a[0]))
            computed_embed_batches.append(output_a.data.cpu().numpy())
            computed_embed_labels.append(a[1].data.cpu().numpy())

    return np.vstack(computed_embed_batches), np.hstack(computed_embed_labels)

def test(output_path: str, data_loading_funcs: dict):

    logger.info(f"Testing Type4Py model")
    logger.info(f"**********************************************************************")
    # Loading dataset
    logger.info("Loading train and test sets...")
    
    # Model's hyper parameters
    model_params = load_model_params()
    train_data_loader, valid_data_loader = load_training_data_per_model(data_loading_funcs, output_path,
                                                                        model_params['batches_test'], train_mode=False)

    model = torch.load(join(output_path, f"type4py_{data_loading_funcs['name']}_model.pt"))
    logger.info(f"Loaded the pre-trained Type4Py {data_loading_funcs['name']} model")

    annoy_index, embed_labels = build_type_clusters(model.model, train_data_loader, valid_data_loader)
    logger.info("Created type clusters")

    annoy_index.save(join(output_path, f"type4py_{data_loading_funcs['name']}_type_cluster"))
    np.save(join(output_path, f"type4py_{data_loading_funcs['name']}_true.npy"), embed_labels)
    logger.info("Saved type clusters")

    test_data_loader, t_idx = load_test_data_per_model(data_loading_funcs, output_path, model_params['batches_test'])
    logger.info("Mapping test samples to type clusters")
    test_type_embed, embed_test_labels = compute_type_embed_batch(model.model, test_data_loader)
    
    # Perform KNN search and predict
    logger.info("Performing KNN search")
    le_all = pickle.load(open(join(output_path, "label_encoder_all.pkl"), 'rb'))
    train_valid_labels = le_all.inverse_transform(embed_labels)
    embed_test_labels = le_all.inverse_transform(embed_test_labels)
    pred_types = predict_type_embed_task(test_type_embed, embed_test_labels,
                                                          train_valid_labels,
                                                          t_idx, annoy_index, model_params['k'])
    
    save_json(join(output_path, f"type4py_{data_loading_funcs['name']}_test_predictions.json"), pred_types)
    logger.info("Saved the Type4Py model's predictions on the disk")
