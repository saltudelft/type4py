from type4py.data_loaders import select_data, TripletDataset
from type4py.learn import load_model_params, TripletModel, create_knn_index
from typing import Tuple
from collections import defaultdict
from os.path import join
from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from annoy import AnnoyIndex
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_types_score(types_dist: list, types_idx: list, types_embed_labels: np.array):
        types_dist = 1 / (np.array(types_dist) + 1e-10) ** 2
        types_dist /= np.sum(types_dist)
        types_score = defaultdict(int)
        for n, d in zip(types_idx, types_dist):
            types_score[types_embed_labels[n]] += d
        
        return sorted({t: s for t, s in types_score.items()}.items(), key=lambda kv: kv[1],
                      reverse=True)

def predict_type_embed(types_embed_array: np.array, types_embed_labels: np.array, indexed_knn: AnnoyIndex,
                       k: int) -> np.array:
    """
    Predict type of given type embedding vectors
    """

    pred_types_embed = []
    pred_types_score = []
    for embed_vec in tqdm(types_embed_array, total=len(types_embed_array)):
        idx, dist = indexed_knn.get_nns_by_vector(embed_vec, k, include_distances=True)
        pred_idx_scores = compute_types_score(dist, idx, types_embed_labels)
        pred_types_embed.append([i for i, s in pred_idx_scores])
        pred_types_score.append(pred_idx_scores)
    
    return np.array(pred_types_embed), pred_types_score

def compute_type_embed_batch(model: TripletModel, data_loader: DataLoader) -> Tuple[np.array, np.array]:
    """
    Compute type embeddings for the whole dataset
    """

    computed_embed_batches = []
    computed_embed_labels = []

    for batch_i, (a, p, n) in enumerate(tqdm(data_loader, total=len(data_loader))):
        model.eval()
        with torch.no_grad():
            output_a = model(*(s.to(DEVICE) for s in a[0]))
            computed_embed_batches.append(output_a.data.cpu().numpy())
            computed_embed_labels.append(a[1].data.cpu().numpy())

    return np.vstack(computed_embed_batches), np.hstack(computed_embed_labels)

def test(output_path: str, data_loading_funcs: dict):

    # Loading dataset
    load_data_t = time()
    X_id_train, X_tok_train, X_type_train = data_loading_funcs['train'](output_path)
    X_id_test, X_tok_test, X_type_test = data_loading_funcs['test'](output_path)
    X_id_valid, X_tok_valid, X_type_valid = data_loading_funcs['valid'](output_path)
    Y_all_train, Y_all_valid, Y_all_test = data_loading_funcs['labels'](output_path)
    print("Loaded train and test sets in %.2f min" % ((time()-load_data_t) / 60))

    print(f"Number of test samples: {len(X_id_test):,}")

    # Select data points which has at least frequency of 3 or more (for similary learning)
    train_mask = select_data(Y_all_train, 3)
    X_id_train, X_tok_train, X_type_train, Y_all_train = X_id_train[train_mask], \
                X_tok_train[train_mask], X_type_train[train_mask], Y_all_train[train_mask]

    valid_mask = select_data(Y_all_valid, 3)
    X_id_valid, X_tok_valid, X_type_valid, Y_all_valid = X_id_valid[valid_mask], \
                X_tok_valid[valid_mask], X_type_valid[valid_mask], Y_all_valid[valid_mask]


    # Model's hyper parameters
    model_params = load_model_params()

    # Batch loaders
    train_loader = DataLoader(TripletDataset(X_id_train, X_tok_train, X_type_train, \
                          labels=Y_all_train, dataset_name=data_loading_funcs['name'], train_mode=True), \
                          batch_size=model_params['batches'], shuffle=True, pin_memory=True,
                          num_workers=4)
    valid_loader = DataLoader(TripletDataset(X_id_valid, X_tok_valid, X_type_valid, \
                            labels=Y_all_valid, dataset_name=data_loading_funcs['name'], \
                                            train_mode=True), batch_size=model_params['batches'], num_workers=4)
    test_loader = DataLoader(TripletDataset(X_id_test, X_tok_test, X_type_test, \
                             labels=Y_all_test, dataset_name=data_loading_funcs['name'],
                              train_mode=False), batch_size=model_params['batches'])

    model = torch.load(join(output_path, f"type4py_{data_loading_funcs['name']}_model.pt"))

    # Create Type Clusters
    train_type_embed, embed_train_labels = compute_type_embed_batch(model.model, train_loader)
    valid_type_embed, embed_valid_labels = compute_type_embed_batch(model.model, valid_loader)
    test_type_embed, embed_test_labels = compute_type_embed_batch(model.model, test_loader)
    annoy_index = create_knn_index(train_type_embed, valid_type_embed, train_type_embed.shape[1], 20)

    # Perform KNN search and predict
    pred_test_embed, pred_test_score = predict_type_embed(test_type_embed,
                                                          np.concatenate((embed_train_labels, embed_valid_labels)),
                                                          annoy_index, model_params['k'])