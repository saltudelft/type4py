from typing import Tuple
from os.path import join
from collections import Counter
from torch.utils.data import TensorDataset
import torch
import numpy as np


def load_data_tensors_TW(filename, limit=-1):
    return torch.from_numpy(np.load(filename)).float()

def load_flat_labels_tensors(filename):

    return torch.from_numpy(np.load(filename)).long()

def load_combined_train_data(output_path: str):
    return torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_param_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'identifiers_ret_train_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_param_train_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'tokens_ret_train_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'train', 'params_train_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'train', 'ret_train_aval_types_dp.npy'))))

def load_combined_valid_data(output_path: str):
    return torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_param_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'identifiers_ret_valid_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_param_valid_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'tokens_ret_valid_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'params_valid_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'valid', 'ret_valid_aval_types_dp.npy'))))

def load_combined_test_data(output_path: str):
    return torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_param_test_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'test', 'identifiers_ret_test_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_param_test_datapoints_x.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'test', 'tokens_ret_test_datapoints_x.npy')))), \
           torch.cat((load_data_tensors_TW(join(output_path, 'vectors', 'test', 'params_test_aval_types_dp.npy')),
                      load_data_tensors_TW(join(output_path, 'vectors', 'test', 'ret_test_aval_types_dp.npy'))))

def load_combined_labels(output_path: str):
    return torch.cat((load_flat_labels_tensors(join(output_path, 'vectors', 'train', 'params_train_dps_y_all.npy')),
                      load_flat_labels_tensors(join(output_path, 'vectors', 'train', 'ret_train_dps_y_all.npy')))), \
           torch.cat((load_flat_labels_tensors(join(output_path, 'vectors', 'valid', 'params_valid_dps_y_all.npy')),
                      load_flat_labels_tensors(join(output_path, 'vectors', 'valid', 'ret_valid_dps_y_all.npy')))), \
           torch.cat((load_flat_labels_tensors(join(output_path, 'vectors', 'test', 'params_test_dps_y_all.npy')),
                      load_flat_labels_tensors(join(output_path, 'vectors', 'test', 'ret_test_dps_y_all.npy'))))




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
