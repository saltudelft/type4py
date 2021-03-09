from type4py.data_loaders import select_data, TripletDataset
from type4py.vectorize import AVAILABLE_TYPES_NUMBER, W2V_VEC_LENGTH
from type4py.eval import eval_type_embed
from type4py.utils import load_json
from type4py import logger
from torch.utils.data import DataLoader
from typing import Tuple
from collections import Counter
from os.path import join
from time import time
from annoy import AnnoyIndex
import numpy as np
import torch.nn as nn
import torch
import pickle

logger.name = __name__
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_params(params_file_path: str=None) -> dict:

    if params_file_path is not None:
        logger.info("Loading user-provided hyper-parameters for the Type4Py model...")
        return load_json(params_file_path)
    else:
        return {'epochs': 10, 'lr': 0.002, 'dr': 0.25, 'output_size': 4096,
                'batches': 2536, 'layers': 1, 'hidden_size': 512,
                'margin': 2.0, 'k': 10}

class Type4Py(nn.Module):
 
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
        return self.model(*(s.to(DEVICE) for s in a)), \
               self.model(*(s.to(DEVICE) for s in p)), \
               self.model(*(s.to(DEVICE) for s in n))

def create_knn_index(train_types_embed: np.array, valid_types_embed: np.array, type_embed_dim:int,
                     tree_size: int) -> AnnoyIndex:
    """
    Creates KNNs index for given type embedding vectors
    """

    annoy_idx = AnnoyIndex(type_embed_dim, 'euclidean')

    for i, v in enumerate(train_types_embed):
        annoy_idx.add_item(i, v)

    if valid_types_embed is not None:
        for i, v in enumerate(valid_types_embed):
            annoy_idx.add_item(len(train_types_embed) + i, v)

    annoy_idx.build(tree_size)
    return annoy_idx

def train_loop_dsl(model: TripletModel, criterion, optimizer, train_data_loader: DataLoader,
                   valid_data_loader: DataLoader, learning_rate: float, epochs: int,
                   common_types: set, model_path: str):
    from type4py.predict import predict_type_embed
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start_t = time()
        total_loss = 0

        for batch_i, (anchor, positive_ex, negative_ex) in enumerate(train_data_loader):

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
         
        logger.info(f"epoch: {epoch} train loss: {total_loss} in {(time() - epoch_start_t) / 60.0:.2f} min.")

        if epoch % 5 == 0:
            valid_start = time()
            valid_loss, valid_all_acc = compute_validation_loss_dsl(model, criterion, train_data_loader, valid_data_loader,
                                                                    predict_type_embed, common_types)
            logger.info(f"epoch: {epoch} valid loss: {valid_loss} in {(time() - valid_start) / 60.0:.2f} min.")
            #torch.save(model.module, join(model_path, f"{model.module.tw_embed_model.__class__.__name__}_{train_data_loader.dataset.dataset_name}_e{epoch}_{datetime.now().strftime('%b%d_%H-%M-%S')}.pt"))

def compute_validation_loss_dsl(model: TripletModel, criterion, train_valid_loader: DataLoader,
                                valid_data_loader: DataLoader, pred_func: callable,
                                 common_types: set) -> Tuple[float, float]:
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

        for batch_i, (a, p, n) in enumerate(train_valid_loader):
            #a_id, a_tok, a_cm, a_avl = a[0]
            output_a = main_model_forward(*(s.to(DEVICE) for s in a[0]))
            computed_embed_batches_train.append(output_a.data.cpu().numpy())
            computed_embed_labels_train.append(a[1].data.cpu().numpy())
        
        for batch_i, (anchor, positive_ex, negative_ex) in enumerate(valid_data_loader):
            positive_ex, _ = positive_ex[0], positive_ex[1]
            negative_ex, _ = negative_ex[0], negative_ex[1]

            anchor_embed, positive_ex_embed, negative_ex_embed = model(anchor[0], positive_ex, negative_ex)
            loss = criterion(anchor_embed, positive_ex_embed, negative_ex_embed)
            valid_total_loss += loss.item()

            output_a = main_model_forward(*(s.to(DEVICE) for s in anchor[0]))
            computed_embed_batches_valid.append(output_a.data.cpu().numpy())
            computed_embed_labels_valid.append(anchor[1].data.cpu().numpy())

        annoy_index = create_knn_index(np.vstack(computed_embed_batches_train), None, computed_embed_batches_train[0].shape[1], 20)
        pred_valid_embed, _ = pred_func(np.vstack(computed_embed_batches_valid), np.hstack(computed_embed_labels_train),
                                                                annoy_index, 10)
        acc_all, acc_common, acc_rare, _, _ = eval_type_embed(pred_valid_embed, np.hstack(computed_embed_labels_valid),
                                                           common_types, 10)
        logger.info("E-All: %.2f | E-Comm: %.2f | E-rare: %.2f" % (acc_all, acc_common, acc_rare))

    return valid_total_loss, acc_all

def train(output_path: str, data_loading_funcs: dict, model_params_path=None):
    
    logger.info(f"Training Type4Py model for {data_loading_funcs['name']} prediction task")
    logger.info(f"***********************************************************************")
    # Loading dataset
    load_data_t = time()
    X_id_train, X_tok_train, X_type_train = data_loading_funcs['train'](output_path)
    X_id_valid, X_tok_valid, X_type_valid = data_loading_funcs['valid'](output_path)
    Y_all_train, Y_all_valid, _ = data_loading_funcs['labels'](output_path)
    logger.info("Loaded train and valid sets in %.2f min" % ((time()-load_data_t) / 60))

    logger.info(f"No. of training samples: {len(X_id_train):,}")
    logger.info(f"No. of validation samples: {len(X_id_valid):,}")

    # Select data points which has at least frequency of 3 or more (for similarity learning)
    train_mask = select_data(Y_all_train, 3)
    X_id_train, X_tok_train, X_type_train, Y_all_train = X_id_train[train_mask], \
                X_tok_train[train_mask], X_type_train[train_mask], Y_all_train[train_mask]

    valid_mask = select_data(Y_all_valid, 3)
    X_id_valid, X_tok_valid, X_type_valid, Y_all_valid = X_id_valid[valid_mask], \
                X_tok_valid[valid_mask], X_type_valid[valid_mask], Y_all_valid[valid_mask]

    count_types = Counter(Y_all_train.data.numpy())
    common_types = [t.item() for t in Y_all_train if count_types[t.item()] >= 100]
    logger.info("Percentage of common types: %.2f%%" % (len(common_types) / Y_all_train.shape[0]*100.0))
    common_types = set(common_types)

    with open(join(output_path, f"{data_loading_funcs['name']}_common_types.pkl"), 'wb') as f:
        pickle.dump(common_types, f)

    # Model's hyper parameters
    model_params = load_model_params(model_params_path)

    # Batch loaders
    train_loader = DataLoader(TripletDataset(X_id_train, X_tok_train, X_type_train, \
                          labels=Y_all_train, dataset_name=data_loading_funcs['name'], train_mode=True), \
                          batch_size=model_params['batches'], shuffle=True, pin_memory=True,
                          num_workers=4)
    valid_loader = DataLoader(TripletDataset(X_id_valid, X_tok_valid, X_type_valid, \
                            labels=Y_all_valid, dataset_name=data_loading_funcs['name'], \
                                            train_mode=True), batch_size=model_params['batches'], num_workers=4)

    # Loading the model
    model = Type4Py(W2V_VEC_LENGTH, model_params['hidden_size'], AVAILABLE_TYPES_NUMBER, model_params['layers'],
                    model_params['output_size'], model_params['dr']).to(DEVICE)
    model = TripletModel(model).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = torch.nn.TripletMarginLoss(margin=model_params['margin'])
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['lr'])

    train_t = time()
    train_loop_dsl(model, criterion, optimizer, train_loader, valid_loader, model_params['lr'],
                   model_params['epochs'], common_types, None)
    logger.info("Training finished in %.2f min" % ((time()-train_t) / 60))

    # Saving the model
    logger.info("Saved the trained Type4Py model for %s prediction on the disk" % data_loading_funcs['name'])
    torch.save(model.module if torch.cuda.device_count() > 1 else model, join(output_path, f"type4py_{data_loading_funcs['name']}_model.pt"))
