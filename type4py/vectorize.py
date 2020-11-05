from gensim.models import Word2Vec
from time import time
from tqdm import tqdm
from type4py.utils import mk_dir_not_exist
import os
import multiprocessing
import numpy as np
import pandas as pd

tqdm.pandas()

W2V_VEC_LENGTH = 100
AVAILABLE_TYPES_NUMBER = 1024

class TokenIterator:
    def __init__(self, param_df: pd.DataFrame, return_df: pd.DataFrame) -> None:
        self.param_df = param_df
        self.return_df = return_df

    def __iter__(self):
        for return_expr_sentences in self.return_df['return_expr_str'][self.return_df['return_expr_str'] != '']:
            yield return_expr_sentences.split()

        for code_occur_sentences in self.param_df['arg_occur'][self.param_df['arg_occur'] != '']:
            yield code_occur_sentences.split()

        for func_name_sentences in self.param_df['func_name'][~self.param_df['func_name'].isnull()]:
            yield func_name_sentences.split()

        for arg_names_sentences in self.return_df['arg_names_str'][self.return_df['arg_names_str'] != '']:
            yield arg_names_sentences.split()

class W2VEmbedding:
    """
    Word2Vec embeddings for code tokens and identifiers
    """
    def __init__(self, param_df: pd.DataFrame, return_df: pd.DataFrame, w2v_model_tk_path) -> None:
        self.param_df = param_df
        self.return_df = return_df
        self.w2v_model_tk_path = w2v_model_tk_path

    def train_model(self, corpus_iterator: TokenIterator, model_path_name: str) -> None:
        """
        Train a Word2Vec model and save the output to a file.
        :param corpus_iterator: class that can provide an iterator that goes through the corpus
        :param model_path_name: path name of the output file
        """

        cores = multiprocessing.cpu_count()

        w2v_model = Word2Vec(min_count=5,
                             window=5,
                             size=W2V_VEC_LENGTH,
                             workers=cores-1)

        t = time()

        w2v_model.build_vocab(sentences=corpus_iterator)

        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        t = time()

        w2v_model.train(sentences=corpus_iterator,
                        total_examples=w2v_model.corpus_count,
                        epochs=20,
                        report_delay=1)

        print('Time to train model: {} mins'.format(round((time() - t) / 60, 2)))

        w2v_model.save(model_path_name)

    def train_token_model(self):
        """
        Trains a W2V model for tokens.
        """

        self.train_model(TokenIterator(self.param_df, self.return_df), self.w2v_model_tk_path)

def vectorize_sequence(sequence: str, feature_length: int, w2v_model: Word2Vec) -> np.ndarray:
    """
    Vectorize a sequence to a multi-dimensial NumPy array
    """

    vector = np.zeros((feature_length, W2V_VEC_LENGTH))

    for i, word in enumerate(sequence.split()):
        if i >= feature_length:
            break
        try:
            vector[i] = w2v_model.wv[word]
        except KeyError:
            pass

    return vector

class IdentifierSequence:
    """
    Vector representation of identifiers
    """
    def __init__(self, identifiers_embd, arg_name, args_name, func_name):

        self.identifiers_embd = identifiers_embd
        self.arg_name = arg_name
        self.args_name = args_name
        self.func_name = func_name

    def seq_length_param(self):

        return {
            "arg_name": 10,
            "sep": 1,
            "func_name": 10,
            "args_name": 10
        }

    def seq_length_return(self):

        return {
            "func_name": 10,
            "sep": 1,
            "args_name": 10,
            "padding": 10
        }

    def generate_datapoint(self, seq_length):
        datapoint = np.zeros((sum(seq_length.values()), W2V_VEC_LENGTH))
        separator = np.ones(W2V_VEC_LENGTH)

        p = 0
        for seq, length in seq_length.items():
            if seq == "sep":
                datapoint[p] = separator
                p += 1
            elif seq == 'padding':
                for i in range(0, length):
                    datapoint[p] = np.zeros(W2V_VEC_LENGTH)
                    p += 1
            else:
                try:
                    for w in vectorize_sequence(self.__dict__[seq], length, self.identifiers_embd):
                        datapoint[p] = w
                        p += 1
                except AttributeError:
                    pass

        return datapoint

    def param_datapoint(self):

        return self.generate_datapoint(self.seq_length_param())

    def return_datapoint(self):

        return self.generate_datapoint(self.seq_length_return())


class TokenSequence:
    """
    Vector representation of code tokens
    """
    def __init__(self, token_model, len_tk_seq, num_tokens_seq, args_usage, return_expr):
        self.token_model = token_model
        self.len_tk_seq = len_tk_seq
        self.num_tokens_seq = num_tokens_seq
        self.args_usage = args_usage
        self.return_expr = return_expr

    def param_datapoint(self):
        datapoint = np.zeros((self.num_tokens_seq*self.len_tk_seq, W2V_VEC_LENGTH))
        for i, w in enumerate(vectorize_sequence(self.args_usage, self.num_tokens_seq*self.len_tk_seq, self.token_model)):
            datapoint[i] = w

        return datapoint

    def return_datapoint(self):
        datapoint = np.zeros((self.num_tokens_seq*self.len_tk_seq, W2V_VEC_LENGTH))

        if isinstance(self.return_expr, str):
            p = 0
            for w in vectorize_sequence(self.return_expr, self.len_tk_seq, self.token_model):
                datapoint[p] = w
                p += 1

        return datapoint

def process_datapoints(f_name, output_path, embedding_type, type, trans_func, cached_file: bool=False):

    if not os.path.exists(os.path.join(output_path, embedding_type + type + '_datapoints_x.npy')) or not cached_file:
        df = pd.read_csv(f_name, na_filter=False)
        datapoints = df.apply(trans_func, axis=1)

        datapoints_X = np.stack(datapoints.progress_apply(lambda x: x.return_datapoint() if 'ret' in type else x.param_datapoint()),
                                axis=0)
        np.save(os.path.join(output_path, embedding_type + type + '_datapoints_x'), datapoints_X)

        return datapoints_X
    else:
        print(f"file {embedding_type + type + '_datapoints_x'} exists!")
        return

def type_vector(size, index):
    v = np.zeros(size)
    v[index] = 1
    return v

def gen_aval_types_datapoints(df_params, df_ret, set_type, output_path, cached_file: bool=False):
    """
    It generates data points for available types.
    """

    if not (os.path.exists(os.path.join(output_path, f'params_{set_type}_aval_types_dp.npy')) and os.path.exists(os.path.join(output_path,
            f'ret_{set_type}_aval_types_dp.npy'))) or not cached_file:

        df_params = pd.read_csv(df_params)
        df_ret = pd.read_csv(df_ret)

        aval_types_params = np.stack(df_params.progress_apply(lambda row: type_vector(AVAILABLE_TYPES_NUMBER, row.param_aval_enc),
                                                    axis=1), axis=0)
        aval_types_ret = np.stack(df_ret.progress_apply(lambda row: type_vector(AVAILABLE_TYPES_NUMBER, row.ret_aval_enc),
                                            axis=1), axis=0)

        np.save(os.path.join(output_path, f'params_{set_type}_aval_types_dp'), aval_types_params)
        np.save(os.path.join(output_path, f'ret_{set_type}_aval_types_dp'), aval_types_ret)

        return aval_types_params, aval_types_ret
    else:
        print(f'file params_{set_type}_aval_types_dp.npy exists!')
        print(f'file ret_{set_type}_aval_types_dp.npy exists!')
        return None, None

def gen_labels_vector(params_df: pd.DataFrame, returns_df: pd.DataFrame, set_type: str, output_path: str):
    """
    It generates a flattened labels vector
    """

    params_df = pd.read_csv(params_df)
    returns_df = pd.read_csv(returns_df)

    np.save(os.path.join(output_path, f'params_{set_type}_dps_y_all'), params_df['arg_type_enc_all'].values)
    np.save(os.path.join(output_path, f'ret_{set_type}_dps_y_all'), returns_df['return_type_enc_all'].values)

    return params_df['arg_type_enc_all'].values, returns_df['return_type_enc_all'].values

def vectorize_args_ret(output_path: str):
    """
    Creates vector representation of functions' arguments and return values
    """

    param_df = pd.read_csv(os.path.join(output_path, "_ml_param_train.csv"), na_filter=False)
    return_df = pd.read_csv(os.path.join(output_path, "_ml_ret_train.csv"), na_filter=False)

    print(f"Number of parameters types: {param_df.shape[0]:,}")
    print(f"Number of returns types: {return_df.shape[0]:,}")

    embedder = W2VEmbedding(param_df, return_df, os.path.join(output_path, 'w2v_token_model.bin'))
    embedder.train_token_model()

    w2v_token_model = Word2Vec.load(os.path.join(output_path, 'w2v_token_model.bin'))
    print(f"W2V token model vocab size : {len(w2v_token_model.wv.vocab):,}")

    # Create dirs for vectors
    mk_dir_not_exist(os.path.join(output_path, "vectors"))
    mk_dir_not_exist(os.path.join(output_path, "vectors", "train"))
    mk_dir_not_exist(os.path.join(output_path, "vectors", "valid"))
    mk_dir_not_exist(os.path.join(output_path, "vectors", "test"))

    tks_seq_len = (7, 3)
    vts_seq_len = (15, 5)
    # Vectorize functions' arguments
    id_trans_func_param = lambda row: IdentifierSequence(w2v_token_model, row.arg_name, row.other_args, row.func_name)
    token_trans_func_param = lambda row: TokenSequence(w2v_token_model, tks_seq_len[0], tks_seq_len[1], row.arg_occur, None)

    # Identifiers
    print("[arg][identifiers] Generating vectors")
    process_datapoints(os.path.join(output_path, "_ml_param_train.csv"),
                       os.path.join(output_path, "vectors", "train"),
                       'identifiers_', 'param_train', id_trans_func_param)
    process_datapoints(os.path.join(output_path, "_ml_param_valid.csv"),
                       os.path.join(output_path, "vectors", "valid"),
                       'identifiers_', 'param_valid', id_trans_func_param)
    process_datapoints(os.path.join(output_path, "_ml_param_test.csv"),
                       os.path.join(output_path, "vectors", "test"),
                       'identifiers_', 'param_test', id_trans_func_param)
    
    # Tokens
    print("[arg][code tokens] Generating vectors")
    process_datapoints(os.path.join(output_path, "_ml_param_train.csv"),
                       os.path.join(output_path, "vectors", "train"),
                       'tokens_', 'param_train', token_trans_func_param)
    process_datapoints(os.path.join(output_path, "_ml_param_valid.csv"),
                       os.path.join(output_path, "vectors", "valid"),
                       'tokens_', 'param_valid', token_trans_func_param)
    process_datapoints(os.path.join(output_path, "_ml_param_test.csv"),
                       os.path.join(output_path, "vectors", "test"),
                       'tokens_', 'param_test', token_trans_func_param)

    # Vectorize functions' return types
    id_trans_func_ret = lambda row: IdentifierSequence(w2v_token_model, None, row.arg_names_str, row.name)
    token_trans_func_ret = lambda row: TokenSequence(w2v_token_model, tks_seq_len[0], tks_seq_len[1], None, row.return_expr_str)

    # Identifiers
    print("[ret][identifiers] Generating vectors")
    process_datapoints(os.path.join(output_path, "_ml_ret_train.csv"),
                       os.path.join(output_path, "vectors", "train"),
                       'identifiers_', 'ret_train', id_trans_func_ret)
    process_datapoints(os.path.join(output_path, "_ml_ret_valid.csv"),
                       os.path.join(output_path, "vectors", "valid"),
                       'identifiers_', 'ret_valid', id_trans_func_ret)
    process_datapoints(os.path.join(output_path, "_ml_ret_test.csv"),
                       os.path.join(output_path, "vectors", "test"),
                       'identifiers_', 'ret_test', id_trans_func_ret)

    # Tokens
    print("[ret][code tokens] Generating vectors")
    process_datapoints(os.path.join(output_path, "_ml_ret_train.csv"),
                       os.path.join(output_path, "vectors", "train"),
                       'tokens_', 'ret_train', token_trans_func_ret)
    process_datapoints(os.path.join(output_path, "_ml_ret_valid.csv"),
                       os.path.join(output_path, "vectors", "valid"),
                       'tokens_', 'ret_valid', token_trans_func_ret)
    process_datapoints(os.path.join(output_path, "_ml_ret_test.csv"),
                       os.path.join(output_path, "vectors", "test"),
                       'tokens_', 'ret_test', token_trans_func_ret)

    # Generate data points for visible type hints
    print("[visible type hints] Generating vectors")
    gen_aval_types_datapoints(os.path.join(output_path, "_ml_param_train.csv"),
                              os.path.join(output_path, "_ml_ret_train.csv"),
                              'train', os.path.join(output_path, "vectors", "train"))
    gen_aval_types_datapoints(os.path.join(output_path, "_ml_param_valid.csv"),
                              os.path.join(output_path, "_ml_ret_valid.csv"),
                              'valid', os.path.join(output_path, "vectors", "valid"))
    gen_aval_types_datapoints(os.path.join(output_path, "_ml_param_test.csv"),
                              os.path.join(output_path, "_ml_ret_test.csv"),
                              'test', os.path.join(output_path, "vectors", "test"))

    # a flattened vector for labels
    print("[true labels] Generating vectors")
    gen_labels_vector(os.path.join(output_path, "_ml_param_train.csv"),
                      os.path.join(output_path, "_ml_ret_train.csv"),
                      'train', os.path.join(output_path, "vectors", "train"))
    gen_labels_vector(os.path.join(output_path, "_ml_param_valid.csv"),
                      os.path.join(output_path, "_ml_ret_valid.csv"),
                      'valid', os.path.join(output_path, "vectors", "valid"))
    gen_labels_vector(os.path.join(output_path, "_ml_param_test.csv"),
                      os.path.join(output_path, "_ml_ret_test.csv"),
                      'test', os.path.join(output_path, "vectors", "test"))
    

