from gensim.models import Word2Vec
from time import time
from tqdm import tqdm
from type4py import logger, AVAILABLE_TYPES_NUMBER, TOKEN_SEQ_LEN
from type4py.utils import mk_dir_not_exist
import os
import multiprocessing
import numpy as np
import pandas as pd

logger.name = __name__
tqdm.pandas()

W2V_VEC_LENGTH = 100

class TokenIterator:
    def __init__(self, param_df: pd.DataFrame, return_df: pd.DataFrame,
                 var_df: pd.DataFrame) -> None:
        self.param_df = param_df
        self.return_df = return_df
        self.var_df = var_df

    def __iter__(self):
        for return_expr_sentences in self.return_df['return_expr_str'][self.return_df['return_expr_str'] != '']:
            yield return_expr_sentences.split()

        for code_occur_sentences in self.param_df['arg_occur'][self.param_df['arg_occur'] != '']:
            yield code_occur_sentences.split()

        for func_name_sentences in self.param_df['func_name'][~self.param_df['func_name'].isnull()]:
            yield func_name_sentences.split()

        for arg_names_sentences in self.return_df['arg_names_str'][self.return_df['arg_names_str'] != '']:
            yield arg_names_sentences.split()

        for var_names_sentences in self.var_df['var_name'][self.var_df['var_name'].notnull()]:
            yield var_names_sentences.split()

        for var_occur_sentences in self.var_df['var_occur'][self.var_df['var_occur'] != '']:
            yield var_occur_sentences.split()

class W2VEmbedding:
    """
    Word2Vec embeddings for code tokens and identifiers
    """
    def __init__(self, param_df: pd.DataFrame, return_df: pd.DataFrame,
                 var_df: pd.DataFrame, w2v_model_tk_path) -> None:
        self.param_df = param_df
        self.return_df = return_df
        self.var_df = var_df
        self.w2v_model_tk_path = w2v_model_tk_path

    def train_model(self, corpus_iterator: TokenIterator, model_path_name: str) -> None:
        """
        Train a Word2Vec model and save the output to a file.
        :param corpus_iterator: class that can provide an iterator that goes through the corpus
        :param model_path_name: path name of the output file
        """

        w2v_model = Word2Vec(min_count=5,
                             window=5,
                             vector_size=W2V_VEC_LENGTH,
                             workers=multiprocessing.cpu_count())

        t = time()
        w2v_model.build_vocab(corpus_iterable=corpus_iterator)
        logger.info('Built W2V vocab in {} mins'.format(round((time() - t) / 60, 2)))
        logger.info(f"W2V model's vocab size: {len(w2v_model.wv):,}")

        t = time()
        w2v_model.train(corpus_iterable=corpus_iterator,
                        total_examples=w2v_model.corpus_count,
                        epochs=20,
                        report_delay=1)

        logger.info('Built W2V model in {} mins'.format(round((time() - t) / 60, 2)))
        w2v_model.save(model_path_name)

    def train_token_model(self):
        """
        Trains a W2V model for tokens.
        """

        self.train_model(TokenIterator(self.param_df, self.return_df, self.var_df), self.w2v_model_tk_path)

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
    def __init__(self, identifiers_embd, arg_name, args_name, func_name, var_name):

        self.identifiers_embd = identifiers_embd
        self.arg_name = arg_name
        self.args_name = args_name
        self.func_name = func_name
        self.var_name = var_name

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

    def seq_length_var(self):
        return {
            'var_name': 10,
            "padding": 21
        }

    def __gen_datapoint(self, seq_length):
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

    # def param_datapoint(self):
    #     return self.__gen_datapoint(self.seq_length_param())

    # def return_datapoint(self):
    #     return self.__gen_datapoint(self.seq_length_return())

    # def var_datapoint(self):
    #     return self.__gen_datapoint(self.seq_length_var())

    def generate_datapoint(self):
        if self.arg_name is not None and self.args_name is not None and self.func_name is None:
            return self.__gen_datapoint(self.seq_length_param())
        elif self.args_name is not None and self.func_name is not None:
            return self.__gen_datapoint(self.seq_length_return())
        elif self.var_name is not None:
            return self.__gen_datapoint(self.seq_length_var())

class TokenSequence:
    """
    Vector representation of code tokens
    """
    def __init__(self, token_model, len_tk_seq, num_tokens_seq, args_usage,
                 return_expr, var_usage):
        self.token_model = token_model
        self.len_tk_seq = len_tk_seq
        self.num_tokens_seq = num_tokens_seq
        self.args_usage = args_usage
        self.return_expr = return_expr
        self.var_usage = var_usage

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

    def var_datapoint(self):
        datapoint = np.zeros((self.num_tokens_seq*self.len_tk_seq, W2V_VEC_LENGTH))
        for i, w in enumerate(vectorize_sequence(self.var_usage, self.num_tokens_seq*self.len_tk_seq, self.token_model)):
            datapoint[i] = w
        return datapoint

    def generate_datapoint(self):
        if self.args_usage is not None:
            return self.param_datapoint()
        elif self.return_expr is not None:
            return self.return_datapoint()
        elif self.var_usage is not None:
            return self.var_datapoint()

def process_datapoints(df, output_path, embedding_type, type, trans_func, cached_file: bool=True):

    if not os.path.exists(os.path.join(output_path, embedding_type + type + '_datapoints_x.npy')) or not cached_file:
        datapoints = df.apply(trans_func, axis=1)

        datapoints_X = np.stack(datapoints.progress_apply(lambda x: x.generate_datapoint()),
                                axis=0)
        np.save(os.path.join(output_path, embedding_type + type + '_datapoints_x'), datapoints_X)

        return datapoints_X
    else:
        logger.warn(f"file {embedding_type + type + '_datapoints_x'} exists!")
        return

def type_vector(size, index):
    v = np.zeros(size)
    v[index] = 1
    return v

def gen_aval_types_datapoints(df_params, df_ret, df_var, set_type, output_path, cached_file: bool=False):
    """
    It generates data points for available types.
    """

    if not (os.path.exists(os.path.join(output_path, f'params_{set_type}_aval_types_dp.npy')) and os.path.exists(os.path.join(output_path,
            f'ret_{set_type}_aval_types_dp.npy'))) and os.path.exists(os.path.join(output_path, f'var_{set_type}_aval_types_dp.npy')) \
                 or not cached_file:

        aval_types_params = np.stack(df_params.progress_apply(lambda row: type_vector(AVAILABLE_TYPES_NUMBER, row.param_aval_enc),
                                                    axis=1), axis=0)
        aval_types_ret = np.stack(df_ret.progress_apply(lambda row: type_vector(AVAILABLE_TYPES_NUMBER, row.ret_aval_enc),
                                            axis=1), axis=0)
        aval_types_var = np.stack(df_var.progress_apply(lambda row: type_vector(AVAILABLE_TYPES_NUMBER, row.var_aval_enc),
                                            axis=1), axis=0)

        np.save(os.path.join(output_path, f'params_{set_type}_aval_types_dp'), aval_types_params)
        np.save(os.path.join(output_path, f'ret_{set_type}_aval_types_dp'), aval_types_ret)
        np.save(os.path.join(output_path, f'var_{set_type}_aval_types_dp'), aval_types_var)

        return aval_types_params, aval_types_ret, aval_types_var
    else:
        logger.warn(f'file params_{set_type}_aval_types_dp.npy exists!')
        logger.warn(f'file ret_{set_type}_aval_types_dp.npy exists!')
        logger.warn(f'file var_{set_type}_aval_types_dp.npy exists!')
        return None, None, None

def gen_labels_vector(params_df: pd.DataFrame, returns_df: pd.DataFrame, var_df: pd.DataFrame,
                      set_type: str, output_path: str):
    """
    It generates a flattened labels vector
    """
    np.save(os.path.join(output_path, f'params_{set_type}_dps_y_all'), params_df['arg_type_enc_all'].values)
    np.save(os.path.join(output_path, f'ret_{set_type}_dps_y_all'), returns_df['return_type_enc_all'].values)
    np.save(os.path.join(output_path, f'var_{set_type}_dps_y_all'), var_df['var_type_enc_all'].values)

    return params_df['arg_type_enc_all'].values, returns_df['return_type_enc_all'].values, var_df['var_type_enc_all'].values

def vectorize_args_ret(output_path: str):
    """
    Creates vector representation of functions' arguments and return values
    """

    train_param_df = pd.read_csv(os.path.join(output_path, "_ml_param_train.csv"), na_filter=False)
    train_return_df = pd.read_csv(os.path.join(output_path, "_ml_ret_train.csv"), na_filter=False)
    train_var_df = pd.read_csv(os.path.join(output_path, "_ml_var_train.csv"), na_filter=False)
    logger.info("Loaded the training data")

    valid_param_df = pd.read_csv(os.path.join(output_path, "_ml_param_valid.csv"), na_filter=False)
    valid_return_df = pd.read_csv(os.path.join(output_path, "_ml_ret_valid.csv"), na_filter=False)
    valid_var_df = pd.read_csv(os.path.join(output_path, "_ml_var_valid.csv"), na_filter=False)
    logger.info("Loaded the validation data")

    test_param_df = pd.read_csv(os.path.join(output_path, "_ml_param_test.csv"), na_filter=False)
    test_return_df = pd.read_csv(os.path.join(output_path, "_ml_ret_test.csv"), na_filter=False)
    test_var_df = pd.read_csv(os.path.join(output_path, "_ml_var_test.csv"), na_filter=False)
    logger.info("Loaded the test data")

    if not os.path.exists(os.path.join(output_path, 'w2v_token_model.bin')):
        embedder = W2VEmbedding(train_param_df, train_return_df, train_var_df,
                                os.path.join(output_path, 'w2v_token_model.bin'))
        embedder.train_token_model()
    else:
        logger.warn("Loading an existing pre-trained W2V model!")

    w2v_token_model = Word2Vec.load(os.path.join(output_path, 'w2v_token_model.bin'))

    # Create dirs for vectors
    mk_dir_not_exist(os.path.join(output_path, "vectors"))
    mk_dir_not_exist(os.path.join(output_path, "vectors", "train"))
    mk_dir_not_exist(os.path.join(output_path, "vectors", "valid"))
    mk_dir_not_exist(os.path.join(output_path, "vectors", "test"))

    #tks_seq_len = (7, 3)
    vts_seq_len = (15, 5)
    # Vectorize functions' arguments
    id_trans_func_param = lambda row: IdentifierSequence(w2v_token_model, row.arg_name, row.other_args,
                                                         row.func_name, None)
    token_trans_func_param = lambda row: TokenSequence(w2v_token_model, TOKEN_SEQ_LEN[0], TOKEN_SEQ_LEN[1],
                                                       row.arg_occur, None, None)

    # Identifiers
    logger.info("[arg][identifiers] Generating vectors")
    process_datapoints(train_param_df,
                       os.path.join(output_path, "vectors", "train"),
                       'identifiers_', 'param_train', id_trans_func_param)
    process_datapoints(valid_param_df,
                       os.path.join(output_path, "vectors", "valid"),
                       'identifiers_', 'param_valid', id_trans_func_param)
    process_datapoints(test_param_df,
                       os.path.join(output_path, "vectors", "test"),
                       'identifiers_', 'param_test', id_trans_func_param)
    
    # Tokens
    logger.info("[arg][code tokens] Generating vectors")
    process_datapoints(train_param_df,
                       os.path.join(output_path, "vectors", "train"),
                       'tokens_', 'param_train', token_trans_func_param)
    process_datapoints(valid_param_df,
                       os.path.join(output_path, "vectors", "valid"),
                       'tokens_', 'param_valid', token_trans_func_param)
    process_datapoints(test_param_df,
                       os.path.join(output_path, "vectors", "test"),
                       'tokens_', 'param_test', token_trans_func_param)

    # Vectorize functions' return types
    id_trans_func_ret = lambda row: IdentifierSequence(w2v_token_model, None, row.arg_names_str, row.name, None)
    token_trans_func_ret = lambda row: TokenSequence(w2v_token_model, TOKEN_SEQ_LEN[0], TOKEN_SEQ_LEN[1], None,
                                                     row.return_expr_str, None)

    # Identifiers
    logger.info("[ret][identifiers] Generating vectors")
    process_datapoints(train_return_df,
                       os.path.join(output_path, "vectors", "train"),
                       'identifiers_', 'ret_train', id_trans_func_ret)
    process_datapoints(valid_return_df,
                       os.path.join(output_path, "vectors", "valid"),
                       'identifiers_', 'ret_valid', id_trans_func_ret)
    process_datapoints(test_return_df,
                       os.path.join(output_path, "vectors", "test"),
                       'identifiers_', 'ret_test', id_trans_func_ret)

    # Tokens
    logger.info("[ret][code tokens] Generating vectors")
    process_datapoints(train_return_df,
                       os.path.join(output_path, "vectors", "train"),
                       'tokens_', 'ret_train', token_trans_func_ret)
    process_datapoints(valid_return_df,
                       os.path.join(output_path, "vectors", "valid"),
                       'tokens_', 'ret_valid', token_trans_func_ret)
    process_datapoints(test_return_df,
                       os.path.join(output_path, "vectors", "test"),
                       'tokens_', 'ret_test', token_trans_func_ret)

    # Vectorize variables types
    id_trans_func_var = lambda row: IdentifierSequence(w2v_token_model, None, None, None, row.var_name)
    token_trans_func_var = lambda row: TokenSequence(w2v_token_model, TOKEN_SEQ_LEN[0], TOKEN_SEQ_LEN[1], None,
                                                     None, row.var_occur)

    # Identifiers
    logger.info("[var][identifiers] Generating vectors")
    process_datapoints(train_var_df,
                       os.path.join(output_path, "vectors", "train"),
                       'identifiers_', 'var_train', id_trans_func_var)
    process_datapoints(valid_var_df,
                       os.path.join(output_path, "vectors", "valid"),
                       'identifiers_', 'var_valid', id_trans_func_var)
    process_datapoints(test_var_df,
                       os.path.join(output_path, "vectors", "test"),
                       'identifiers_', 'var_test', id_trans_func_var)

    # Tokens
    logger.info("[var][code tokens] Generating vectors")
    process_datapoints(train_var_df,
                       os.path.join(output_path, "vectors", "train"),
                       'tokens_', 'var_train', token_trans_func_var)
    process_datapoints(valid_var_df,
                       os.path.join(output_path, "vectors", "valid"),
                       'tokens_', 'var_valid', token_trans_func_var)
    process_datapoints(test_var_df,
                       os.path.join(output_path, "vectors", "test"),
                       'tokens_', 'var_test', token_trans_func_var)
    
    # Generate data points for visible type hints
    logger.info("[visible type hints] Generating vectors")
    gen_aval_types_datapoints(train_param_df, train_return_df, train_var_df,
                              'train', os.path.join(output_path, "vectors", "train"))
    gen_aval_types_datapoints(valid_param_df, valid_return_df, valid_var_df,
                              'valid', os.path.join(output_path, "vectors", "valid"))
    gen_aval_types_datapoints(test_param_df, test_return_df, test_var_df,
                              'test', os.path.join(output_path, "vectors", "test"))

    # a flattened vector for labels
    logger.info("[true labels] Generating vectors")
    gen_labels_vector(train_param_df, train_return_df, train_var_df,
                      'train', os.path.join(output_path, "vectors", "train"))
    gen_labels_vector(valid_param_df, valid_return_df, valid_var_df,
                      'valid', os.path.join(output_path, "vectors", "valid"))
    gen_labels_vector(test_param_df, test_return_df, test_var_df,
                      'test', os.path.join(output_path, "vectors", "test"))
    

