import os

from type4py.data_loaders import select_data, TripletDataset, load_training_data_per_model, \
    load_training_data_per_model_sep
from type4py.vectorize import AVAILABLE_TYPES_NUMBER, W2V_VEC_LENGTH
from type4py.eval import eval_type_embed
from type4py.utils import load_model_params
from type4py import logger, MIN_DATA_POINTS, KNN_TREE_SIZE
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
import pkg_resources

logger.name = __name__
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelNotFit(Exception):
    pass

class NotCompleteModel(ModelNotFit):
    def __init__(self):
        super().__init__("learn_split may just fit for complete model!")

class TrainedModel(Exception):
    pass

class ModelTrainedError(TrainedModel):
    def __init__(self):
        super().__init__("Model has been trained for this dataset!")

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


def load_model(model_type: str, model_params: dict):
    """
    Load the Type4Py model with desired confings
    """

    if model_type == "complete":
        return Type4Py(W2V_VEC_LENGTH, model_params['hidden_size'], AVAILABLE_TYPES_NUMBER, model_params['layers'],
                       model_params['output_size'], model_params['dr']).to(DEVICE)
    else:
        raise NotCompleteModel


def create_knn_index(train_types_embed: np.array, valid_types_embed: np.array, type_embed_dim: int) -> AnnoyIndex:
    """
    Creates KNNs index for given type embedding vectors
    """

    annoy_idx = AnnoyIndex(type_embed_dim, 'euclidean')

    for i, v in enumerate(tqdm(train_types_embed, total=len(train_types_embed),
                               desc="KNN index")):
        annoy_idx.add_item(i, v)

    if valid_types_embed is not None:
        for i, v in enumerate(valid_types_embed):
            annoy_idx.add_item(len(train_types_embed) + i, v)

    annoy_idx.build(KNN_TREE_SIZE)
    return annoy_idx


def train_loop_dsl(model: TripletModel, criterion, optimizer, train_data_loader: DataLoader,
                   valid_data_loader: DataLoader, learning_rate: float, epochs: int,
                   ubiquitous_types: str, common_types: set, model_path: str):
    from type4py.predict import predict_type_embed

    for epoch in range(1, epochs + 1):
        model.train()
        # epoch_start_t = time()
        total_loss = 0

        for batch_i, (anchor, positive_ex, negative_ex) in enumerate(tqdm(train_data_loader,
                                                                          total=len(train_data_loader),
                                                                          desc=f"Epoch {epoch}")):
            anchor, _ = anchor[0], anchor[1]
            positive_ex, _ = positive_ex[0], positive_ex[1]
            negative_ex, _ = negative_ex[0], negative_ex[1]

            optimizer.zero_grad()
            anchor_embed, positive_ex_embed, negative_ex_embed = model(anchor, positive_ex, negative_ex)
            loss = criterion(anchor_embed, positive_ex_embed, negative_ex_embed)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"epoch: {epoch} train loss: {total_loss}")

        if valid_data_loader is not None:
            if epoch % 5 == 0:
                logger.info("Evaluating on validation set")
                valid_start = time()
                valid_loss, valid_all_acc = compute_validation_loss_dsl(model, criterion, train_data_loader,
                                                                        valid_data_loader,
                                                                        predict_type_embed, ubiquitous_types,
                                                                        common_types)
                logger.info(f"epoch: {epoch} valid loss: {valid_loss} in {(time() - valid_start) / 60.0:.2f} min.")
                # torch.save(model.module, join(model_path, f"{model.module.tw_embed_model.__class__.__name__}_{train_data_loader.dataset.dataset_name}_e{epoch}_{datetime.now().strftime('%b%d_%H-%M-%S')}.pt"))


def compute_validation_loss_dsl(model: TripletModel, criterion, train_valid_loader: DataLoader,
                                valid_data_loader: DataLoader, pred_func: callable,
                                ubiquitous_types: str, common_types: set) -> Tuple[float, float]:
    """
    Computes validation loss for Deep Similarity Learning-based approach
    """

    valid_total_loss = 0
    with torch.no_grad():
        model.eval()

        if isinstance(model, nn.DataParallel):
            main_model_forward = model.module.model
        else:
            main_model_forward = model.model

        computed_embed_batches_train = []
        computed_embed_labels_train = []
        computed_embed_batches_valid = []
        computed_embed_labels_valid = []

        for batch_i, (anchor, positive_ex, negative_ex) in enumerate(tqdm(valid_data_loader,
                                                                          total=len(valid_data_loader),
                                                                          desc="Type Cluster - Valid set")):
            positive_ex, _ = positive_ex[0], positive_ex[1]
            negative_ex, _ = negative_ex[0], negative_ex[1]

            anchor_embed, positive_ex_embed, negative_ex_embed = model(anchor[0], positive_ex, negative_ex)
            loss = criterion(anchor_embed, positive_ex_embed, negative_ex_embed)
            valid_total_loss += loss.item()

            output_a = main_model_forward(*(s.to(DEVICE) for s in anchor[0]))
            computed_embed_batches_valid.append(output_a.data.cpu().numpy())
            computed_embed_labels_valid.append(anchor[1].data.cpu().numpy())

    return valid_total_loss, 0.0

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

def train_split(output_path: str, data_loading_funcs: dict, dataset_type: str,  model_params_path=None, validation: bool = False):
    logger.info(f"Training Type4Py model")
    logger.info(f"***********************************************************************")

    # Model's hyper parameters
    model_params = load_model_params(model_params_path)
    data_type_list = ["var", "param", "ret"]
    if dataset_type not in data_type_list:
        raise ValueError(f"{dataset_type} is not in the default data type list!")

    train_data_loader, valid_data_loader = load_training_data_per_model_sep(data_loading_funcs, output_path,dataset_type,
                                                                        model_params['batches'],
                                                                        load_valid_data=validation,
                                                                        no_workers=cpu_count() // 2)

    # Loading label encoder and finding ubiquitous & common types
    le_all = pickle.load(open(join(output_path, "label_encoder_all.pkl"), 'rb'))
    count_types = Counter(train_data_loader.dataset.labels.data.numpy())

    var_exists, param_exits, ret_exists = check_pickle_file(dataset_type, data_loading_funcs, output_path)

    if os.path.exists(join(output_path, f"{data_loading_funcs['name']}_common_types_{dataset_type}.pkl")):
        logger.warn(f"{data_loading_funcs['name']}_common_types_{dataset_type}.pkl file exists!")

    with open(join(output_path, f"{data_loading_funcs['name']}_common_types_{dataset_type}.pkl"), 'wb') as f:
        pickle.dump(count_types, f)

    type_filename = dataset_type

    if var_exists and dataset_type != "var":
        with open(join(output_path, f"{data_loading_funcs['name']}_common_types_var.pkl"), 'rb') as f1:
            count_types_var = pickle.load(f1)
        count_types.update(count_types_var)
        type_filename = type_filename + "_var"

    if param_exits and dataset_type != "param":
        with open(join(output_path, f"{data_loading_funcs['name']}_common_types_param.pkl"), 'rb') as f2:
            count_types_param = pickle.load(f2)
        count_types.update(count_types_param)
        type_filename = type_filename + "_param"

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

    criterion = torch.nn.TripletMarginLoss(margin=model_params['margin'])
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['lr'])

    train_t = time()
    train_loop_dsl(model, criterion, optimizer, train_data_loader,
                   valid_data_loader if validation else None, model_params['lr'],
                   model_params['epochs'], ubiquitous_types, common_types, None)
    logger.info("Training finished in %.2f min" % ((time() - train_t) / 60))

    # Saving the model
    logger.info("Saved the trained Type4Py model for %s prediction on the disk" % data_loading_funcs['name'])
    os.remove(output_path, trained_model_name)
    torch.save(model.module if torch.cuda.device_count() > 1 else model,
               join(output_path, f"{trained_model_name[:-3]}_{dataset_type}.pt"))
