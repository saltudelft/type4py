from type4py import logger, MIN_DATA_POINTS
from typing import Tuple
from os.path import join
from collections import Counter
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

logger.name = __name__

def load_data_tensors_TW(filename, limit=-1):
    return torch.from_numpy(np.load(filename)).float()

def load_flat_labels_tensors(filename):

    return torch.from_numpy(np.load(filename)).long()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Combined data
def load_combined_train_data(output_path: str):
    return torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_param_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_ret_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_var_train_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_param_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_ret_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_var_train_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'train', 'params_train_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'ret_train_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'var_train_aval_types_dp.npy'))))
   
def load_combined_valid_data(output_path: str):
    return torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_param_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_ret_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_var_valid_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_param_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_ret_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_var_valid_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'params_valid_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'ret_valid_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'var_valid_aval_types_dp.npy'))))

def load_combined_test_data(output_path: str):
    id_p_te = load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_param_test_datapoints_x.npy'))
    id_r_te = load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_ret_test_datapoints_x.npy'))
    id_v_te = load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_var_test_datapoints_x.npy'))

    return torch.cat((id_p_te, id_r_te, id_v_te)), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_param_test_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_ret_test_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_var_test_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'test', 'params_test_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'test', 'ret_test_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'test', 'var_test_aval_types_dp.npy')))), \
           (len(id_p_te)-1, (len(id_p_te)+len(id_r_te))-1, (len(id_p_te)+len(id_r_te)+len(id_v_te))-1)
           # indexes of combined test data for separating between prediction tasks
           
def load_combined_labels(output_path: str):
    return torch.cat((load_flat_labels_tensors(join(output_path, 'vectors', 'train', 'params_train_dps_y_all.npy')),
                      load_flat_labels_tensors(join(output_path, 'vectors', 'train', 'ret_train_dps_y_all.npy')),
                      load_flat_labels_tensors(join(output_path, 'vectors', 'train', 'var_train_dps_y_all.npy')))), \
           torch.cat((load_flat_labels_tensors(join(output_path, 'vectors', 'valid', 'params_valid_dps_y_all.npy')),
                      load_flat_labels_tensors(join(output_path, 'vectors', 'valid', 'ret_valid_dps_y_all.npy')),
                      load_flat_labels_tensors(join(output_path, 'vectors', 'valid', 'var_valid_dps_y_all.npy')))), \
           torch.cat((load_flat_labels_tensors(join(output_path, 'vectors', 'test', 'params_test_dps_y_all.npy')),
                      load_flat_labels_tensors(join(output_path, 'vectors', 'test', 'ret_test_dps_y_all.npy')),
                      load_flat_labels_tensors(join(output_path, 'vectors', 'test', 'var_test_dps_y_all.npy'))))

# Loading data for Type4Py model w/o identifiers
def load_combined_train_data_woi(output_path: str):
    return torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_param_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_ret_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_var_train_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'train', 'params_train_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'ret_train_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'var_train_aval_types_dp.npy'))))

def load_combined_valid_data_woi(output_path: str):
    return torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_param_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_ret_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_var_valid_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'params_valid_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'ret_valid_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'var_valid_aval_types_dp.npy'))))

def load_combined_test_data_woi(output_path: str):
    tk_p_te = load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_param_test_datapoints_x.npy'))
    tk_r_te = load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_ret_test_datapoints_x.npy'))
    tk_v_te = load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_var_test_datapoints_x.npy'))

    return torch.cat((tk_p_te, tk_r_te, tk_v_te)), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'test', 'params_test_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'test', 'ret_test_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'test', 'var_test_aval_types_dp.npy')))), \
           (len(tk_p_te)-1, (len(tk_p_te)+len(tk_r_te))-1, (len(tk_p_te)+len(tk_r_te)+len(tk_v_te))-1)
           # indexes of combined test data for separating between prediction tasks

# Loading data for Type4Py model w/o code contexts
def load_combined_train_data_woc(output_path: str):
    return torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_param_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_ret_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_var_train_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'train', 'params_train_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'ret_train_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'var_train_aval_types_dp.npy'))))

def load_combined_valid_data_woc(output_path: str):
    return torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_param_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_ret_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_var_valid_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'params_valid_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'ret_valid_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'var_valid_aval_types_dp.npy'))))

def load_combined_test_data_woc(output_path: str):
    id_p_te = load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_param_test_datapoints_x.npy'))
    id_r_te = load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_ret_test_datapoints_x.npy'))
    id_v_te = load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_var_test_datapoints_x.npy'))

    return torch.cat((id_p_te, id_r_te, id_v_te)), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'test', 'params_test_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'test', 'ret_test_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'test', 'var_test_aval_types_dp.npy')))), \
           (len(id_p_te)-1, (len(id_p_te)+len(id_r_te))-1, (len(id_p_te)+len(id_r_te)+len(id_v_te))-1)
           # indexes of combined test data for separating between prediction tasks

# Loading data for Type4Py model w/o visible type hints
def load_combined_train_data_wov(output_path: str):
    return torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_param_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_ret_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_var_train_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_param_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_ret_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_var_train_datapoints_x.npy'))))

def load_combined_valid_data_wov(output_path: str):
    return torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_param_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_ret_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_var_valid_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_param_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_ret_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_var_valid_datapoints_x.npy'))))

def load_combined_test_data_wov(output_path: str):
    id_p_te = load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_param_test_datapoints_x.npy'))
    id_r_te = load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_ret_test_datapoints_x.npy'))
    id_v_te = load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_var_test_datapoints_x.npy'))

    return torch.cat((id_p_te, id_r_te, id_v_te)), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_param_test_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_ret_test_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_var_test_datapoints_x.npy')))), \
           (len(id_p_te)-1, (len(id_p_te)+len(id_r_te))-1, (len(id_p_te)+len(id_r_te)+len(id_v_te))-1)
           # indexes of combined test data for separating between prediction tasks


# Argument data
def load_param_train_data(output_path: str):
    return load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_param_train_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_param_train_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'train', 'params_train_aval_types_dp.npy'))

def load_param_valid_data(output_path: str):
    return load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_param_valid_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_param_valid_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'params_valid_aval_types_dp.npy'))

def load_param_test_data(output_path: str):
    return load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_param_test_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_param_test_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'test', 'params_test_aval_types_dp.npy'))

def load_param_labels(output_path: str):
    return load_flat_labels_tensors(join(output_path, 'vectors', 'train', 'params_train_dps_y_all.npy')), \
           load_flat_labels_tensors(join(output_path, 'vectors', 'valid', 'params_valid_dps_y_all.npy')), \
           load_flat_labels_tensors(join(output_path, 'vectors', 'test', 'params_test_dps_y_all.npy'))
                     
# Return data
def load_ret_train_data(output_path: str):
    return load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_ret_train_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_ret_train_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'train', 'ret_train_aval_types_dp.npy'))

def load_ret_valid_data(output_path: str):
    return load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_ret_valid_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_ret_valid_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'ret_valid_aval_types_dp.npy'))

def load_ret_test_data(output_path: str):
    return load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_ret_test_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_ret_test_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'test', 'ret_test_aval_types_dp.npy'))

def load_ret_labels(output_path: str):
    return load_flat_labels_tensors(join(output_path, 'vectors', 'train', 'ret_train_dps_y_all.npy')), \
           load_flat_labels_tensors(join(output_path, 'vectors', 'valid', 'ret_valid_dps_y_all.npy')), \
           load_flat_labels_tensors(join(output_path, 'vectors', 'test', 'ret_test_dps_y_all.npy'))

# Variable data
def load_var_train_data(output_path: str):
    return load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_var_train_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_var_train_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'train', 'var_train_aval_types_dp.npy'))

def load_var_valid_data(output_path: str):
    return load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_var_valid_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_var_valid_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'var_valid_aval_types_dp.npy'))

def load_var_test_data(output_path: str):
    return load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_var_test_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_var_test_datapoints_x.npy')), \
           load_data_tensors_TW(join(output_path, 'vectors', 'test', 'var_test_aval_types_dp.npy'))

def load_var_labels(output_path: str):
    return load_flat_labels_tensors(join(output_path, 'vectors', 'train', 'var_train_dps_y_all.npy')), \
           load_flat_labels_tensors(join(output_path, 'vectors', 'valid', 'var_valid_dps_y_all.npy')), \
           load_flat_labels_tensors(join(output_path, 'vectors', 'test', 'var_test_dps_y_all.npy'))

def select_data(data, n):
    """
    Selects data points that are frequent more than n times
    """
    
    mask = torch.tensor([False] * data.shape[0], dtype=torch.bool)
    counter = Counter(data.data.numpy())
    
    for i, d in enumerate(data):
        if counter[d.item()] >= n:
            mask[i] = True
    
    return mask

def load_training_data_per_model(data_loading_funcs: dict, output_path: str,
                                 no_batches: int, train_mode:bool=True, load_valid_data:bool=True,
                                 no_workers:int=8) -> Tuple[DataLoader, DataLoader]:
    """
    Loads appropriate training data based on the model's type
    """

    # def find_common_types(y_all_train: torch.Tensor):
    #     count_types = Counter(y_all_train.data.numpy())
    #     return [t.item() for t in y_all_train if count_types[t.item()] >= 100]

    load_data_t = time()
    if data_loading_funcs['name'] == 'woi':
        # without identifiers
        X_tok_train, X_type_train = data_loading_funcs['train'](output_path)
        X_tok_valid, X_type_valid = data_loading_funcs['valid'](output_path)
        Y_all_train, Y_all_valid, _ = data_loading_funcs['labels'](output_path)

        train_mask = select_data(Y_all_train, MIN_DATA_POINTS)
        X_tok_train, X_type_train, Y_all_train = X_tok_train[train_mask], \
             X_type_train[train_mask], Y_all_train[train_mask]

        valid_mask = select_data(Y_all_valid, MIN_DATA_POINTS)
        X_tok_valid, X_type_valid, Y_all_valid = X_tok_valid[valid_mask], \
             X_type_valid[valid_mask], Y_all_valid[valid_mask]

        triplet_data_train = TripletDataset(X_tok_train, X_type_train, labels=Y_all_train,
                                      dataset_name=data_loading_funcs['name'], train_mode=train_mode)
        triplet_data_valid = TripletDataset(X_tok_valid, X_type_valid, labels=Y_all_valid,
                                            dataset_name=data_loading_funcs['name'],
                                            train_mode=train_mode)
    
    elif data_loading_funcs['name'] == 'woc':
        # without code tokens
        X_id_train, X_type_train = data_loading_funcs['train'](output_path)
        X_id_valid, X_type_valid = data_loading_funcs['valid'](output_path)
        Y_all_train, Y_all_valid, _ = data_loading_funcs['labels'](output_path)

        train_mask = select_data(Y_all_train, MIN_DATA_POINTS)
        X_id_train, X_type_train, Y_all_train = X_id_train[train_mask], \
                    X_type_train[train_mask], Y_all_train[train_mask]

        valid_mask = select_data(Y_all_valid, MIN_DATA_POINTS)
        X_id_valid, X_type_valid, Y_all_valid = X_id_valid[valid_mask], \
                X_type_valid[valid_mask], Y_all_valid[valid_mask]

        triplet_data_train = TripletDataset(X_id_train, X_type_train, labels=Y_all_train,
                                      dataset_name=data_loading_funcs['name'], train_mode=train_mode)
        triplet_data_valid = TripletDataset(X_id_valid, X_type_valid, labels=Y_all_valid,
                                            dataset_name=data_loading_funcs['name'],
                                            train_mode=train_mode)

    elif data_loading_funcs['name'] == 'wov':
        # without visible type hints
        X_id_train, X_tok_train, = data_loading_funcs['train'](output_path)
        X_id_valid, X_tok_valid, = data_loading_funcs['valid'](output_path)
        Y_all_train, Y_all_valid, _ = data_loading_funcs['labels'](output_path)

        train_mask = select_data(Y_all_train, MIN_DATA_POINTS)
        X_id_train, X_tok_train, Y_all_train = X_id_train[train_mask], \
                    X_tok_train[train_mask], Y_all_train[train_mask]

        valid_mask = select_data(Y_all_valid, MIN_DATA_POINTS)
        X_id_valid, X_tok_valid, Y_all_valid = X_id_valid[valid_mask], \
                    X_tok_valid[valid_mask], Y_all_valid[valid_mask]

        triplet_data_train = TripletDataset(X_id_train, X_tok_train, labels=Y_all_train,
                                      dataset_name=data_loading_funcs['name'], train_mode=train_mode)
        triplet_data_valid = TripletDataset(X_id_valid, X_tok_valid, labels=Y_all_valid,
                                            dataset_name=data_loading_funcs['name'],
                                            train_mode=train_mode)
        
    else:
        # Complete model
        X_id_train, X_tok_train, X_type_train = data_loading_funcs['train'](output_path)
        if load_valid_data:
            Y_all_train, Y_all_valid, _ = data_loading_funcs['labels'](output_path)
        else:
            Y_all_train, _, _ = data_loading_funcs['labels'](output_path)

        train_mask = select_data(Y_all_train, MIN_DATA_POINTS)
       
        X_id_train = X_id_train[train_mask]
        X_tok_train = X_tok_train[train_mask]
        X_type_train = X_type_train[train_mask]
        Y_all_train = Y_all_train[train_mask]
       
        # X_id_train, X_tok_train, X_type_train, Y_all_train = X_id_train[train_mask], \
        #             X_tok_train[train_mask], X_type_train[train_mask], Y_all_train[train_mask]

        triplet_data_train = TripletDataset(X_id_train, X_tok_train, X_type_train, labels=Y_all_train,
                                      dataset_name=data_loading_funcs['name'], train_mode=train_mode)

        logger.info(f"Loaded train set of the {data_loading_funcs['name']} dataset in {(time()-load_data_t)/60:.2f} min")
        
        if load_valid_data:
            X_id_valid, X_tok_valid, X_type_valid = data_loading_funcs['valid'](output_path)
            valid_mask = select_data(Y_all_valid, MIN_DATA_POINTS)
            X_id_valid = X_id_valid[valid_mask]
            X_tok_valid = X_tok_valid[valid_mask]
            X_type_valid = X_type_valid[valid_mask]
            Y_all_valid = Y_all_valid[valid_mask]
            triplet_data_valid = TripletDataset(X_id_valid, X_tok_valid, X_type_valid, labels=Y_all_valid,
                                            dataset_name=data_loading_funcs['name'],
                                            train_mode=train_mode)
            logger.info(f"Loaded valid set of the {data_loading_funcs['name']} dataset")

    train_loader = DataLoader(triplet_data_train, batch_size=no_batches, shuffle=True,
                              pin_memory=True, num_workers=no_workers)

    if load_valid_data:
        valid_loader = DataLoader(triplet_data_valid, batch_size=no_batches, num_workers=no_workers)
        return train_loader, valid_loader
    else:
        return train_loader, None

def load_test_data_per_model(data_loading_funcs: dict, output_path: str,
                             no_batches: int, drop_last_batch:bool=False):
    """
    Loads appropriate training data based on the model's type
    """

    load_data_t = time()
    if data_loading_funcs['name'] == 'woi':
        # without identifiers
        X_tok_test, X_type_test, t_idx = data_loading_funcs['test'](output_path)
        _, _, Y_all_test = data_loading_funcs['labels'](output_path)


        triplet_data_test = TripletDataset(X_tok_test, X_type_test, labels=Y_all_test,
                                           dataset_name=data_loading_funcs['name'], train_mode=False)
    
    elif data_loading_funcs['name'] == 'woc':
        # without code tokens
        X_id_test, X_type_test, t_idx = data_loading_funcs['test'](output_path)
        _, _, Y_all_test = data_loading_funcs['labels'](output_path)


        triplet_data_test = TripletDataset(X_id_test, X_type_test, labels=Y_all_test,
                                           dataset_name=data_loading_funcs['name'], train_mode=False)

    elif data_loading_funcs['name'] == 'wov':
        # without visible type hints
        X_id_test, X_tok_test, t_idx = data_loading_funcs['test'](output_path)
        _, _, Y_all_test = data_loading_funcs['labels'](output_path)


        triplet_data_test = TripletDataset(X_id_test, X_tok_test, labels=Y_all_test,
                                           dataset_name=data_loading_funcs['name'], train_mode=False)
        
    else:
        # Complete model
        X_id_test, X_tok_test, X_type_test, t_idx = data_loading_funcs['test'](output_path)
        _, _, Y_all_test = data_loading_funcs['labels'](output_path)


        triplet_data_test = TripletDataset(X_id_test, X_tok_test, X_type_test, labels=Y_all_test,
                                           dataset_name=data_loading_funcs['name'], train_mode=False)


    logger.info(f"Loaded the test set of the {data_loading_funcs['name']} dataset in {(time()-load_data_t)/60:.2f} min")

    return DataLoader(triplet_data_test, batch_size=no_batches, num_workers=12, drop_last=drop_last_batch), t_idx



class TripletDataset(torch.utils.data.Dataset):

    def __init__(self, *in_sequences: torch.Tensor, labels: torch.Tensor, dataset_name: str,
                 train_mode: bool=True):
        self.data = TensorDataset(*in_sequences)
        self.labels = labels
        self.dataset_name = dataset_name
        self.train_mode = train_mode

        self.get_item_func = self.get_item_train if self.train_mode else self.get_item_test

    def get_item_train(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                         Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        It returns three tuples. Each one is a (data, label)
         - The first tuple is (data, label) at the given index
         - The second tuple is similar (data, label) to the given index
         - The third tuple is different (data, label) from the given index 
        """

         # Find a similar datapoint randomly
        mask = self.labels == self.labels[index]
        mask[index] = False # Making sure that the similar pair is NOT the same as the given index
        mask = mask.nonzero()
        a = mask[torch.randint(high=len(mask), size=(1,))][0]

        # Find a different datapoint randomly
        mask = self.labels != self.labels[index]
        mask = mask.nonzero()
        b = mask[torch.randint(high=len(mask), size=(1,))][0]
        
        return (self.data[index], self.labels[index]), (self.data[a.item()], self.labels[a.item()]), \
               (self.data[b.item()], self.labels[b.item()])

    def get_item_test(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], list, list]:
        return (self.data[index], self.labels[index]), [], []
    
    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                         Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
         return self.get_item_func(index)

    def __len__(self) -> int:
        return len(self.data)
