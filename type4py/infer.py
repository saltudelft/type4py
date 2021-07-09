"""
This module loads the pre-trained Type4Py model to infer type annotations for a given source file.
"""

from typing import List, Optional, Tuple
from torch._C import dtype
from type4py import logger, AVAILABLE_TYPES_NUMBER, TOKEN_SEQ_LEN
from type4py.learn import load_model_params
from type4py.predict import compute_types_score
from type4py.preprocess import trans_aval_type
from type4py.vectorize import IdentifierSequence, TokenSequence, type_vector
from type4py.type_check import MypyManager, type_check_single_file
from type4py.utils import create_tmp_file
from libsa4py.cst_extractor import Extractor
from libsa4py.representations import ModuleInfo
from libsa4py.nl_preprocessing import NLPreprocessor
from libsa4py.cst_transformers import TypeAnnotationRemover, TypeApplier
from libsa4py.cst_visitor import Visitor
from libsa4py.utils import load_json, read_file, save_json, write_file
from libcst.metadata import MetadataWrapper, TypeInferenceProvider
from libcst import parse_module
from annoy import AnnoyIndex
from os.path import basename, join, splitext, dirname
from enum import Enum
from gensim.models import Word2Vec
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os
import regex
import torch
import onnxruntime

logger.name = __name__
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_FILE_SUFFIX = "_type4py_typed.py"

class PretrainedType4Py:
    # def __init__(self, type4py_model, type4py_model_params, w2v_model, type_clusters_idx,
    #              type_clusers_labels, label_enc):
    #     self.type4py_model = type4py_model
    #     self.type4py_model_params = type4py_model_params
    #     self.w2v_model = w2v_model
    #     self.type_clusters_idx = type_clusters_idx
    #     self.type_clusters_labels = type_clusers_labels
    #     self.label_enc = label_enc

    def __init__(self, pre_trained_model_path, device='gpu', pre_read_type_cluster=False):
        self.pre_trained_model_path = pre_trained_model_path
        self.pre_read_type_cluster = pre_read_type_cluster
        self.device = device

        self.type4py_model = None
        self.type4py_model_params = None
        self.w2v_model = None
        self.type_clusters_idx = None
        self.type_clusters_labels = None
        self.label_enc = None
        self.vths = None

    def load_pretrained_model(self):
        #self.type4py_model = torch.load(join(self.pre_trained_model_path, f"type4py_complete_model.pt"))
        self.type4py_model = onnxruntime.InferenceSession(join(self.pre_trained_model_path, f"type4py_complete_model.onnx"))
        self.type4py_model_params = load_model_params()
        logger.info(f"Loaded the pre-trained Type4Py model")

        if self.device == 'gpu':
            self.type4py_model.set_providers(['GPUExecutionProvider'])
            logger.info("The model runs on GPU")
        elif self.device == 'cpu':
            self.type4py_model.set_providers(['CPUExecutionProvider'])
            logger.info("The model runs on CPU")

        self.w2v_model = Word2Vec.load(join(self.pre_trained_model_path, 'w2v_token_model.bin'))
        logger.info(f"Loaded the pre-trained W2V model")

        self.vths = pd.read_csv(join(self.pre_trained_model_path, 'MT4Py_VTHs.csv')).head(AVAILABLE_TYPES_NUMBER)
        self.vths = self.vths['Types'].to_list()

        self.type_clusters_idx = AnnoyIndex(self.type4py_model_params['output_size'], 'euclidean')
        self.type_clusters_idx.load(join(self.pre_trained_model_path, "type4py_complete_type_cluster"),
                                    prefault=self.pre_read_type_cluster)
        self.type_clusters_labels = np.load(join(self.pre_trained_model_path, f"type4py_complete_true.npy"))
        self.label_enc = pickle.load(open(join(self.pre_trained_model_path, "label_encoder_all.pkl"), 'rb'))
        logger.info(f"Loaded the Type Clusters")

    def load_pretrained_model_wo_clusters(self):
        self.type4py_model = torch.load(join(self.pre_trained_model_path, f"type4py_complete_model.pt"))
        self.type4py_model_params = load_model_params()
        logger.info(f"Loaded the pre-trained Type4Py model")

        self.w2v_model = Word2Vec.load(join(self.pre_trained_model_path, 'w2v_token_model.bin'))
        logger.info(f"Loaded the pre-trained W2V model")

        # self.type_clusters_idx = AnnoyIndex(self.type4py_model_params['output_size'], 'euclidean')
        # self.type_clusters_idx.load(join(self.pre_trained_model_path, "type4py_complete_type_cluster"))
        self.type_clusters_labels = np.load(join(self.pre_trained_model_path, f"type4py_complete_true.npy"))
        self.label_enc = pickle.load(open(join(self.pre_trained_model_path, "label_encoder_all.pkl"), 'rb'))
        #logger.info(f"Loaded the Type Clusters")

    def load_type_clusters(self):
        self.type_clusters_idx = AnnoyIndex(self.type4py_model_params['output_size'], 'euclidean')
        self.type_clusters_idx.load(join(self.pre_trained_model_path, "type4py_complete_type_cluster"))

# def load_pretrained_model(pre_trained_model_path):
#     """
#     Loads the pre-trained Type4Py model, type clusters and other required files for inference
#     """

#     model = torch.load(join(pre_trained_model_path, f"type4py_complete_model.pt"))
#     model_params = load_model_params()
#     logger.info(f"Loaded the pre-trained Type4Py model")

#     w2v_model = Word2Vec.load(join(pre_trained_model_path, 'w2v_token_model.bin'))
#     logger.info(f"Loaded the pre-trained W2V model")

#     type_cluster_index = AnnoyIndex(model_params['output_size'], 'euclidean')
#     type_cluster_index.load(join(pre_trained_model_path, "type4py_complete_type_cluster"))
#     types_embed_labels = np.load(join(pre_trained_model_path, f"type4py_complete_true.npy"))
#     le_all = pickle.load(open(join(pre_trained_model_path, "label_encoder_all.pkl"), 'rb'))
#     logger.info(f"Loaded the Type Clusters")

#     return PretrainedType4Py(model, model_params, w2v_model, type_cluster_index, types_embed_labels, le_all)

def analyze_src_f(src_f: str, remove_preexisting_type_annot:bool=False) -> ModuleInfo:
    """
    Removes pre-existing type annotations from a source file
    """

    v = Visitor()
    if remove_preexisting_type_annot:
        mw = MetadataWrapper(parse_module(src_f).visit(TypeAnnotationRemover()),
                         cache={TypeInferenceProvider: {'types':[]}})
    else:
        mw = MetadataWrapper(parse_module(src_f),
                         cache={TypeInferenceProvider: {'types':[]}})
    mw.visit(v)

    return ModuleInfo(v.imports, v.module_variables, v.module_variables_use, v.module_vars_ln, v.cls_list,
                      v.fns, '', '', v.module_no_types, v.module_type_annot_cove)

def type_embed_single_dp(model: onnxruntime.InferenceSession, id_dp, code_tks_dp, vth_dp):
    """
    Gives a type embedding for a single test datapoint.
    """
    model_inputs =  {model.get_inputs()[0].name: id_dp.astype(np.float32, copy=False),
                     model.get_inputs()[1].name: code_tks_dp.astype(np.float32, copy=False),
                     model.get_inputs()[2].name: vth_dp.astype(np.float32, copy=False)}
    return model.run(None, model_inputs)[0].reshape(-1,1)

    # model.eval()
    # with torch.no_grad():
    #     return model(*(i.to(DEVICE) for i in input_tensor)).data.cpu().numpy().reshape(-1,1)

def infer_single_dp(type_cluster_idx: AnnoyIndex, k:int, types_embed_labels:np.array,
                    type_embed_vec: np.array):
    """
    Infers a list of likely types for a single test datapoint.
    """
    idx, dist = type_cluster_idx.get_nns_by_vector(type_embed_vec, k, include_distances=True)
    return compute_types_score(dist, idx, types_embed_labels)

def var2vec(var_name: str, var_occur: str, w2v_model, type4py_model) -> np.array:
    """
    Converts a variable to its type embedding
    """
    df_var = pd.DataFrame([[var_name, var_occur, AVAILABLE_TYPES_NUMBER-1]],
                          columns=['var_name', 'var_occur', 'var_aval_enc'])
    id_dp = df_var.apply(lambda row: IdentifierSequence(w2v_model, None, None, None, row.var_name), axis=1)

    id_dp = np.stack(id_dp.apply(lambda x: x.generate_datapoint()), axis=0)

    code_tks_dp = df_var.apply(lambda row: TokenSequence(w2v_model, TOKEN_SEQ_LEN[0], TOKEN_SEQ_LEN[1],
                                row.var_occur, None, None), axis=1)
    code_tks_dp = np.stack(code_tks_dp.apply(lambda x: x.generate_datapoint()), axis=0)
    vth_dp = np.stack(df_var.apply(lambda row: type_vector(AVAILABLE_TYPES_NUMBER, row.var_aval_enc),
                                        axis=1), axis=0)
    
    return type_embed_single_dp(type4py_model, id_dp, code_tks_dp, vth_dp)

def param2vec(w2v_model, type4py_model, vth, *param_hints) -> np.array:
    """
    Converts a function argument to its type embedding
    """
    # TODO: Fix VTH encoding
    df_param = pd.DataFrame([[p for p in param_hints] + [5]],
                          columns=['func_name', 'arg_name', 'other_args', 'arg_occur', 'param_aval_enc'])

    id_dp = df_param.apply(lambda row: IdentifierSequence(w2v_model, row.arg_name, row.other_args,
                                                          row.func_name, None), axis=1)

    id_dp = np.stack(id_dp.apply(lambda x: x.generate_datapoint()), axis=0)

    code_tks_dp = df_param.apply(lambda row: TokenSequence(w2v_model, TOKEN_SEQ_LEN[0], TOKEN_SEQ_LEN[1],
                                                           row.arg_occur, None, None), axis=1)
    code_tks_dp = np.stack(code_tks_dp.apply(lambda x: x.generate_datapoint()), axis=0)
    vth_dp = np.stack(df_param.apply(lambda row: type_vector(AVAILABLE_TYPES_NUMBER, row.param_aval_enc),
                                        axis=1), axis=0)
    
    return type_embed_single_dp(type4py_model, id_dp, code_tks_dp, vth_dp)

def ret2vec(w2v_model, type4py_model, *ret_hints) -> np.array:
    """
    Converts a function return to its type embedding
    """
    df_ret = pd.DataFrame([[r for r in ret_hints] + [AVAILABLE_TYPES_NUMBER-1]],
                          columns=['func_name', 'arg_names', 'ret_expr_seq', 'ret_aval_enc'])
    id_dp = df_ret.apply(lambda row: IdentifierSequence(w2v_model, None, row.arg_names, row.func_name,
                          None), axis=1)

    id_dp = np.stack(id_dp.apply(lambda x: x.generate_datapoint()),
                            axis=0)

    code_tks_dp = df_ret.apply(lambda row: TokenSequence(w2v_model, TOKEN_SEQ_LEN[0], TOKEN_SEQ_LEN[1], None,
                                                         row.ret_expr_seq, None), axis=1)
    code_tks_dp = np.stack(code_tks_dp.apply(lambda x: x.generate_datapoint()), axis=0)
    vth_dp = np.stack(df_ret.apply(lambda row: type_vector(AVAILABLE_TYPES_NUMBER, row.ret_aval_enc),
                                        axis=1), axis=0)
    
    return type_embed_single_dp(type4py_model, id_dp, code_tks_dp, vth_dp)

def apply_inferred_types(in_src_f: str, in_src_f_dict: dict, out_src_f_path: str):
    """
    Applies inffered type annototations to the source file
    """

    f_parsed = MetadataWrapper(parse_module(in_src_f)).visit(TypeApplier(in_src_f_dict, apply_nlp=False))
    write_file(out_src_f_path, f_parsed.code)


def infer_single_file(src_f_ext: dict, pre_trained_m: PretrainedType4Py) -> dict:
    """
    Infers type annotations for the whole source code file
    """

    def infer_preds_score(type_embed: np.array) -> list:
        """
        Gives a list of predictions with its corresponding probability score
        """
        preds = infer_single_dp(pre_trained_m.type_clusters_idx, pre_trained_m.type4py_model_params['k'],
                                   pre_trained_m.type_clusters_labels, type_embed)
        return list(zip(list(pre_trained_m.label_enc.inverse_transform([p for p,s in preds])), [s for p,s in preds]))

    nlp_prep = NLPreprocessor()
    # Storing Type4Py's predictions
    src_f_ext['variables_p'] = {}
    for m_v, m_v_o in zip(src_f_ext['variables'], src_f_ext['mod_var_occur'].values()):
        
        var_type_embed = var2vec(nlp_prep.process_identifier(m_v),
                                 str([nlp_prep.process_sentence(o) for i in m_v_o for o in i]),
                                  pre_trained_m.w2v_model, pre_trained_m.type4py_model)
        # The type of module-level vars
        src_f_ext['variables_p'][m_v] = infer_preds_score(var_type_embed)

    for i, fn in enumerate(src_f_ext['funcs']):
        fn_n = nlp_prep.process_identifier(fn['name'])
        fn_p = [(n, nlp_prep.process_identifier(n), o) for n, o in zip(fn['params'], fn["params_occur"].values()) if n not in {'args', 'kwargs'}]
        fn['params_p'] = {'args': [], 'kwargs': []}
        for o_p, p, p_o in fn_p:
            param_type_embed = param2vec(pre_trained_m.w2v_model, pre_trained_m.type4py_model, pre_trained_m.vths,
                                         fn_n, p, " ".join([p[1] for p in fn_p]),
                                         str([nlp_prep.process_sentence(o) for i in p_o for o in i]))
            # The type of arguments for module-level functions
            src_f_ext['funcs'][i]['params_p'][o_p] = infer_preds_score(param_type_embed)

        # The type of local variables for module-level functions
        fn['variables_p'] = {}
        for fn_v, fn_v_o in zip(fn['variables'], fn['fn_var_occur'].values()):
            var_type_embed = var2vec(nlp_prep.process_identifier(fn_v),
                                    str([nlp_prep.process_sentence(o) for i in fn_v_o for o in i]),
                                    pre_trained_m.w2v_model, pre_trained_m.type4py_model)

            src_f_ext['funcs'][i]['variables_p'][fn_v] = infer_preds_score(var_type_embed)

        # The return type for module-level functions
        if src_f_ext['funcs'][i]['ret_exprs'] != []:
            src_f_ext['funcs'][i]['ret_type_p'] = {}
            ret_type_embed = ret2vec(pre_trained_m.w2v_model, pre_trained_m.type4py_model, fn_n, fn_p,
                                    " ".join([nlp_prep.process_identifier(r.replace('return ', '')) for r in fn['ret_exprs']]))
            
            src_f_ext['funcs'][i]['ret_type_p'] = infer_preds_score(ret_type_embed)
    
    # The type of class-level vars
    for c_i, c in enumerate(src_f_ext['classes']):
        c['variables_p'] = {}
        for c_v, c_v_o in zip(c['variables'], c['cls_var_occur'].values()):
            cls_var_type_embed = var2vec(nlp_prep.process_identifier(c_v),
                                 str([nlp_prep.process_sentence(o) for i in c_v_o for o in i]),
                                 pre_trained_m.w2v_model, pre_trained_m.type4py_model)
  
            src_f_ext['classes'][c_i]['variables_p'][c_v] = infer_preds_score(cls_var_type_embed)

        # The type of arguments for class-level functions
        # TODO: Ignore triavial funcs such as __str__
        for fn_i, fn in enumerate(c['funcs']):
            fn_n = nlp_prep.process_identifier(fn['name'])
            fn_p = [(n, nlp_prep.process_identifier(n), o) for n, o in zip(fn['params'], fn["params_occur"].values()) if n not in {'args', \
                    'kwargs', 'self'}]
            #fn_p = [nlp_prep.process_identifier(n) for n in fn['params'] if n != 'self']
            fn["params_p"] = {'self': [], 'args': [], 'kwargs': []}
            for o_p, p, p_o in fn_p:
                param_type_embed = param2vec(pre_trained_m.w2v_model, pre_trained_m.type4py_model, pre_trained_m.vths,
                                            fn_n, p, " ".join([p[1] for p in fn_p]),
                                            str([nlp_prep.process_sentence(o) for i in p_o for o in i if o != "self"]))
                
                src_f_ext['classes'][c_i]['funcs'][fn_i]['params_p'][o_p] = infer_preds_score(param_type_embed)

            # The type of local variables for class-level functions
            fn['variables_p'] = {}
            for fn_v, fn_v_o in zip(fn['variables'], fn['fn_var_occur'].values()):
                var_type_embed = var2vec(nlp_prep.process_identifier(fn_v),
                                    str([nlp_prep.process_sentence(o) for i in fn_v_o for o in i]),
                                    pre_trained_m.w2v_model, pre_trained_m.type4py_model)

                src_f_ext['classes'][c_i]['funcs'][fn_i]['variables_p'][fn_v] = infer_preds_score(var_type_embed)
                
            # The return type for class-level functions
            if src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_exprs'] != []:
                src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type_p'] = {}
                ret_type_embed = ret2vec(pre_trained_m.w2v_model, pre_trained_m.type4py_model, fn_n, fn_p,
                                        " ".join([regex.sub(r"self\.?", '', nlp_prep.process_identifier(r.replace('return ', ''))) for r in fn['ret_exprs']]))
                
                src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type_p'] = infer_preds_score(ret_type_embed)

    return src_f_ext


class PredictionType(Enum):
    # Case 1: Type4Py's prediction is equal to ground truth
    p_equal_gt = 1
    # Case 2: Type4Py's prediction is NOT equal to ground truth
    p_not_equal_gt = 2
    # Case 3: Type4Py predicts while there is no ground truth
    p_wo_gt = 3


def type_check_pred(src_f_r: str, src_f_o_path: str, src_f_ext: dict,
                    tc: MypyManager, pred: str, true: str) -> Tuple[bool, PredictionType]:
    """
    Type checks a prediction
    """
    apply_inferred_types(src_f_r, src_f_ext, src_f_o_path)
    #print(read_file(src_f_o_path))
    type_checked = type_check_single_file(src_f_o_path, tc)

    if pred == true:
        return type_checked, PredictionType.p_equal_gt
    elif true == "":
        return type_checked, PredictionType.p_wo_gt
    else:
        return type_checked, PredictionType.p_not_equal_gt

def type_check_inferred_types(src_f_ext: dict, src_f_read: str, src_f_o_path):

    mypy_tc = MypyManager('mypy', 20)
    preds_type_checked: Tuple[bool, PredictionType] = []

    for m_v, m_v_t in src_f_ext['variables'].items():
        # The predictions for module-level vars   
        for p, s in src_f_ext['variables_p'][m_v]:
            logger.info(f"Annotating module-level variable {m_v} with {p}")
            src_f_ext['variables'][m_v] = p
            is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, m_v_t)
            preds_type_checked.append((is_tc, p_type))
            if not is_tc:
                src_f_ext['variables'][m_v] = m_v_t
            
    for i, fn in enumerate(src_f_ext['funcs']):
        for p_n, p_t in fn['params'].items():
            # The predictions for arguments for module-level functions
            for p, s in fn['params_p'][p_n]:
                logger.info(f"Annotating function parameter {p_n} with {p}")
                src_f_ext['funcs'][i]['params'][p_n] = p
                is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, p_t)
                preds_type_checked.append((is_tc, p_type))
                if not is_tc:
                    src_f_ext['funcs'][i]['params'][p_n] = p_t

        # The predictions local variables for module-level functions
        for fn_v, fn_v_t in fn['variables'].items():
            for p, s in fn['variables_p'][fn_v]:
                logger.info(f"Annotating function variable {fn_v} with {p}")
                src_f_ext['funcs'][i]['variables'][fn_v] = p
                is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, fn_v_t)
                preds_type_checked.append((is_tc, p_type))
                if not is_tc:
                    src_f_ext['funcs'][i]['variables'][fn_v] = fn_v_t
            
        # The return type for module-level functions
        if src_f_ext['funcs'][i]['ret_exprs'] != []:
            org_t = src_f_ext['funcs'][i]['ret_type']
            for p, s in src_f_ext['funcs'][i]['ret_type_p']:
                logger.info(f"Annotating function {src_f_ext['funcs'][i]['name']} return with {p}")
                src_f_ext['funcs'][i]['ret_type'] = p
                is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, org_t)
                preds_type_checked.append((is_tc, p_type))
                if not is_tc:
                    src_f_ext['funcs'][i]['ret_type'] = org_t

    # The type of class-level vars
    for c_i, c in enumerate(src_f_ext['classes']):
        for c_v, c_v_t in c['variables'].items():
            for p, s in c['variables_p'][c_v]:
                logger.info(f"Annotating class variable {c_v} with {p}")
                src_f_ext['classes'][c_i]['variables'][c_v] = p
                is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, c_v_t)
                preds_type_checked.append((is_tc, p_type))
                if not is_tc:
                    src_f_ext['classes'][c_i]['variables'][c_v] = c_v_t

        # The type of arguments for class-level functions
        for fn_i, fn in enumerate(c['funcs']):
            for p_n, p_t in fn["params"].items():
                for p, s in fn["params_p"][p_n]:
                    logger.info(f"Annotating function parameter {p_n} with {p}")
                    src_f_ext['classes'][c_i]['funcs'][fn_i]['params'][p_n] = p
                    is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, p_t)
                    preds_type_checked.append((is_tc, p_type))
                    if not is_tc:
                        src_f_ext['classes'][c_i]['funcs'][fn_i]['params'][p_n] = p_t

            # The type of local variables for class-level functions
            for fn_v, fn_v_t in fn['variables'].items():
                for p, s in fn['variables_p'][fn_v]:
                    logger.info(f"Annotating function variable {fn_v} with {p}")
                    src_f_ext['classes'][c_i]['funcs'][fn_i]['variables'][fn_v] = p
                    is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, fn_v_t)
                    preds_type_checked.append((is_tc, p_type))
                    if not is_tc:
                        src_f_ext['classes'][c_i]['funcs'][fn_i]['variables'][fn_v] = fn_v_t

            # The return type for class-level functions
            if src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_exprs'] != []:
                org_t = src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type']
                for p, s in src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type_p']:
                    logger.info(f"Annotating function {src_f_ext['classes'][c_i]['funcs'][fn_i]['name']} return with {p}")
                    src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type'] = p
                    is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, org_t)
                    preds_type_checked.append((is_tc, p_type))
                    if not is_tc:
                        src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type'] = org_t
    
    #apply_inferred_types(src_f_read, src_f_ext, src_f_o_path)
    return report_type_check_preds(preds_type_checked)

# def get_type_checked_preds(src_f_ext: dict, src_f_read: str, src_f_o_path):
#     mypy_tc = MypyManager('mypy', 20)


#     for m_v, m_v_t in tqdm(src_f_ext['variables'].items()):
#         # The predictions for module-level vars
#         mod_vars_p = []   
#         for p, s in src_f_ext['variables_p'][m_v]:
#             logger.info(f"Annotating module-level variable {m_v} with {p}")
#             src_f_ext['variables'][m_v] = p
#             is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, m_v_t)
#             if is_tc:
#                 mod_vars_p.append((p, s))
#         src_f_ext['variables_p'][m_v] = mod_vars_p
#         src_f_ext['variables'][m_v] = m_v_t
            
#     for i, fn in tqdm(enumerate(src_f_ext['funcs']), total=len(src_f_ext['funcs']), desc="[module][funcs]"):
#         for p_n, p_t in fn['params'].items():
#             # The predictions for arguments for module-level functions
#             for p, s in fn['params_p'][p_n]:
#                 logger.info(f"Annotating function parameter {p_n} with {p}")
#                 src_f_ext['funcs'][i]['params'][p_n] = p
#                 is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, p_t)
#                 if not is_tc:
#                     src_f_ext['funcs'][i]['params'][p_n] = p_t

#         # The predictions local variables for module-level functions
#         for fn_v, fn_v_t in fn['variables'].items():
#             for p, s in fn['variables_p'][fn_v]:
#                 logger.info(f"Annotating function variable {fn_v} with {p}")
#                 src_f_ext['funcs'][i]['variables'][fn_v] = p
#                 is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, fn_v_t)
#                 preds_type_checked.append((is_tc, p_type))
#                 if not is_tc:
#                     src_f_ext['funcs'][i]['variables'][fn_v] = fn_v_t
            
#         # The return type for module-level functions
#         if src_f_ext['funcs'][i]['ret_exprs'] != []:
#             org_t = src_f_ext['funcs'][i]['ret_type']
#             for p, s in src_f_ext['funcs'][i]['ret_type_p']:
#                 logger.info(f"Annotating function {src_f_ext['funcs'][i]['name']} return with {p}")
#                 src_f_ext['funcs'][i]['ret_type'] = p
#                 is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, org_t)
#                 preds_type_checked.append((is_tc, p_type))
#                 if not is_tc:
#                     src_f_ext['funcs'][i]['ret_type'] = org_t

#     # The type of class-level vars
#     for c_i, c in tqdm(enumerate(src_f_ext['classes']), total=len(src_f_ext['classes']), desc="[module][classes]"):
#         for c_v, c_v_t in c['variables'].items():
#             for p, s in c['variables_p'][c_v]:
#                 logger.info(f"Annotating class variable {c_v} with {p}")
#                 src_f_ext['classes'][c_i]['variables'][c_v] = p
#                 is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, c_v_t)
#                 preds_type_checked.append((is_tc, p_type))
#                 if not is_tc:
#                     src_f_ext['classes'][c_i]['variables'][c_v] = c_v_t

#         # The type of arguments for class-level functions
#         for fn_i, fn in enumerate(c['funcs']):
#             for p_n, p_t in fn["params"].items():
#                 for p, s in fn["params_p"][p_n]:
#                     logger.info(f"Annotating function parameter {p_n} with {p}")
#                     src_f_ext['classes'][c_i]['funcs'][fn_i]['params'][p_n] = p
#                     is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, p_t)
#                     preds_type_checked.append((is_tc, p_type))
#                     if not is_tc:
#                         src_f_ext['classes'][c_i]['funcs'][fn_i]['params'][p_n] = p_t

#             # The type of local variables for class-level functions
#             for fn_v, fn_v_t in fn['variables'].items():
#                 for p, s in fn['variables_p'][fn_v]:
#                     logger.info(f"Annotating function variable {fn_v} with {p}")
#                     src_f_ext['classes'][c_i]['funcs'][fn_i]['variables'][fn_v] = p
#                     is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, fn_v_t)
#                     preds_type_checked.append((is_tc, p_type))
#                     if not is_tc:
#                         src_f_ext['classes'][c_i]['funcs'][fn_i]['variables'][fn_v] = fn_v_t

#             # The return type for class-level functions
#             if src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_exprs'] != []:
#                 org_t = src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type']
#                 for p, s in src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type_p']:
#                     logger.info(f"Annotating function {src_f_ext['classes'][c_i]['funcs'][fn_i]['name']} return with {p}")
#                     src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type'] = p
#                     is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, org_t)
#                     preds_type_checked.append((is_tc, p_type))
#                     if not is_tc:
#                         src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type'] = org_t
    
#     #apply_inferred_types(src_f_read, src_f_ext, src_f_o_path)
#     return report_type_check_preds(preds_type_checked)

# def get_type_slots_preds_file(source_file_path: str) -> list:

#     src_f_read = read_file(source_file_path)
#     src_f_ext = load_json(join(dirname(source_file_path),
#                               splitext(basename(source_file_path))[0]+"_type4py_typed.json"))

#     f_type_slots_preds = []

#     for m_v, m_v_t in tqdm(src_f_ext['variables'].items()):
#         # The predictions for module-level vars   
#         for p, s in src_f_ext['variables_p'][m_v]:
#             src_f_ext['variables'][m_v] = p
#             f_type_slots_preds.append((source_file_path, src_f_read, src_f_ext, ('variables', m_v), m_v_t, p))
            
#     for i, fn in tqdm(enumerate(src_f_ext['funcs']), total=len(src_f_ext['funcs']), desc="[module][funcs]"):
#         for p_n, p_t in fn['params'].items():
#             # The predictions for arguments for module-level functions
#             for p, s in fn['params_p'][p_n]:
#                 src_f_ext['funcs'][i]['params'][p_n] = p
#                 f_type_slots_preds.append((source_file_path, src_f_read, src_f_ext, ('funcs', i, 'params', p_n), p_t, p))
            
#         # The predictions local variables for module-level functions
#         for fn_v, fn_v_t in fn['variables'].items():
#             for p, s in fn['variables_p'][fn_v]:
#                 src_f_ext['funcs'][i]['variables'][fn_v] = p
#                 f_type_slots_preds.append((source_file_path, src_f_read, src_f_ext, ('funcs', i, 'variables', fn_v), fn_v_t, p))
            
#         # The return type for module-level functions
#         if src_f_ext['funcs'][i]['ret_exprs'] != []:
#             org_t = src_f_ext['funcs'][i]['ret_type']
#             for p, s in src_f_ext['funcs'][i]['ret_type_p']:
#                 src_f_ext['funcs'][i]['ret_type'] = p
#                 f_type_slots_preds.append((source_file_path, src_f_read, src_f_ext, ('funcs', i, 'ret_type'), org_t, p))
                
#     # The type of class-level vars
#     for c_i, c in tqdm(enumerate(src_f_ext['classes']), total=len(src_f_ext['classes']), desc="[module][classes]"):
#         for c_v, c_v_t in c['variables'].items():
#             for p, s in c['variables_p'][c_v]:
#                 src_f_ext['classes'][c_i]['variables'][c_v] = p
#                 f_type_slots_preds.append((source_file_path, src_f_read, src_f_ext, ('classes', c_i, 'variables', c_v), c_v_t, p))

#         # The type of arguments for class-level functions
#         for fn_i, fn in enumerate(c['funcs']):
#             for p_n, p_t in fn["params"].items():
#                 for p, s in fn["params_p"][p_n]:
#                     src_f_ext['classes'][c_i]['funcs'][fn_i]['params'][p_n] = p
#                     f_type_slots_preds.append((source_file_path, src_f_read, src_f_ext, ('classes', c_i, 'funcs', fn_i, 'params', p_n), p_t, p))                

#             # The type of local variables for class-level functions
#             for fn_v, fn_v_t in fn['variables'].items():
#                 for p, s in fn['variables_p'][fn_v]:
#                     src_f_ext['classes'][c_i]['funcs'][fn_i]['variables'][fn_v] = p
#                     f_type_slots_preds.append((source_file_path, src_f_read, src_f_ext, ('classes', c_i, 'funcs', fn_i, 'variables', fn_v), fn_v_t, p)) 
                     
#             # The return type for class-level functions
#             if src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_exprs'] != []:
#                 org_t = src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type']
#                 for p, s in src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type_p']:
#                     src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type'] = p
#                     f_type_slots_preds.append((source_file_path, src_f_read, src_f_ext, ('classes', c_i, 'funcs', fn_i, 'ret_type'), org_t, p)) 
    
#     #apply_inferred_types(src_f_read, src_f_ext, src_f_o_path)
#     return f_type_slots_preds

def get_type_checked_preds(src_f_ext: dict, src_f_read: str) -> dict:

    mypy_tc = MypyManager('mypy', 20)
    tmp_f = create_tmp_file(".py")
  
    for m_v, m_v_t in tqdm(src_f_ext['variables'].items()):
        # The predictions for module-level vars
        for p, s in src_f_ext['variables_p'][m_v][:]:
            src_f_ext['variables'][m_v] = p
            is_tc, _ = type_check_pred(src_f_read, tmp_f.name, src_f_ext, mypy_tc, p, m_v_t)
            if not is_tc:
                src_f_ext['variables_p'][m_v].remove((p, s))
            
    for i, fn in tqdm(enumerate(src_f_ext['funcs']), total=len(src_f_ext['funcs']), desc="[module][funcs]"):
        for p_n, p_t in fn['params'].items():
            # The predictions for arguments for module-level functions
            for p, s in fn['params_p'][p_n][:]:
                src_f_ext['funcs'][i]['params'][p_n] = p
                is_tc, _ = type_check_pred(src_f_read, tmp_f.name, src_f_ext, mypy_tc, p, p_t)
                if not is_tc:
                    src_f_ext['funcs'][i]['params_p'][p_n].remove((p, s))

        # The predictions local variables for module-level functions
        for fn_v, fn_v_t in fn['variables'].items():
            for p, s in fn['variables_p'][fn_v][:]:
                src_f_ext['funcs'][i]['variables'][fn_v] = p
                is_tc, _ = type_check_pred(src_f_read, tmp_f.name, src_f_ext, mypy_tc, p, fn_v_t)
                if not is_tc:
                    src_f_ext['funcs'][i]['variables_p'][fn_v].remove((p, s))
            
        # The return type for module-level functions
        if src_f_ext['funcs'][i]['ret_exprs'] != []:
            org_t = src_f_ext['funcs'][i]['ret_type']
            for p, s in src_f_ext['funcs'][i]['ret_type_p'][:]:
                src_f_ext['funcs'][i]['ret_type'] = p
                is_tc, _ = type_check_pred(src_f_read, tmp_f.name, src_f_ext, mypy_tc, p, org_t)
                if not is_tc:
                    src_f_ext['funcs'][i]['ret_type_p'].remove((p, s))

    # The type of class-level vars
    for c_i, c in tqdm(enumerate(src_f_ext['classes']), total=len(src_f_ext['classes']), desc="[module][classes]"):
        for c_v, c_v_t in c['variables'].items():
            for p, s in c['variables_p'][c_v][:]:
                src_f_ext['classes'][c_i]['variables'][c_v] = p
                is_tc, _ = type_check_pred(src_f_read, tmp_f.name, src_f_ext, mypy_tc, p, c_v_t)
                if not is_tc:
                    src_f_ext['classes'][c_i]['variables_p'][c_v].remove((p, s))

        # The type of arguments for class-level functions
        for fn_i, fn in enumerate(c['funcs']):
            for p_n, p_t in fn["params"].items():
                for p, s in fn["params_p"][p_n][:]:
                    src_f_ext['classes'][c_i]['funcs'][fn_i]['params'][p_n] = p
                    is_tc, _ = type_check_pred(src_f_read, tmp_f.name, src_f_ext, mypy_tc, p, p_t)
                    if not is_tc:
                        src_f_ext['classes'][c_i]['funcs'][fn_i]['params_p'][p_n].remove((p, s))

            # The type of local variables for class-level functions
            for fn_v, fn_v_t in fn['variables'].items():
                for p, s in fn['variables_p'][fn_v][:]:
                    src_f_ext['classes'][c_i]['funcs'][fn_i]['variables'][fn_v] = p
                    is_tc, _ = type_check_pred(src_f_read, tmp_f.name, src_f_ext, mypy_tc, p, fn_v_t)
                    if not is_tc:
                        src_f_ext['classes'][c_i]['funcs'][fn_i]['variables_p'][fn_v].remove((p, s))

            # The return type for class-level functions
            if src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_exprs'] != []:
                org_t = src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type']
                for p, s in src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type_p'][:]:
                    src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type'] = p
                    is_tc, _ = type_check_pred(src_f_read, tmp_f.name, src_f_ext, mypy_tc, p, org_t)
                    if not is_tc:
                        src_f_ext['classes'][c_i]['funcs'][fn_i]['ret_type_p'].remove((p, s))
    
    os.unlink(tmp_f.name)
    return src_f_ext


def report_type_check_preds(type_check_preds: List[Tuple[bool, PredictionType]]) -> Tuple[Optional[float],
                                                                                          Optional[float], Optional[float]]:

    no_p_equal_gt = 0
    no_p_equal_gt_tc = 0
    no_p_not_equal_gt = 0
    no_p_not_equal_gt_tc = 0
    no_p_wo_gt = 0
    no_p_wo_gt_tc = 0

    p_equal_gt = None
    p_not_equal_gt = None
    p_wo_gt = None

    for is_tc, p_t in type_check_preds:
        if p_t == PredictionType.p_equal_gt:
            no_p_equal_gt += 1
            if is_tc:
                no_p_equal_gt_tc += 1
        elif p_t == PredictionType.p_not_equal_gt:
            no_p_not_equal_gt += 1
            if is_tc:
                no_p_not_equal_gt_tc += 1
        elif p_t == PredictionType.p_wo_gt:
            no_p_wo_gt += 1
            if is_tc:
                no_p_wo_gt_tc += 1

    if no_p_equal_gt != 0:
        p_equal_gt = no_p_equal_gt_tc / no_p_equal_gt
        logger.info(f"g -> (p==g) {p_equal_gt:.2f}")
    if no_p_not_equal_gt != 0:
        p_not_equal_gt = no_p_not_equal_gt_tc / no_p_not_equal_gt
        logger.info(f"g -> (p!=g) {p_not_equal_gt:.2f}")
    if no_p_wo_gt != 0:
        p_wo_gt = no_p_wo_gt_tc / no_p_wo_gt
        logger.info(f"e -> p {p_wo_gt:.2f}")

    return p_equal_gt, p_not_equal_gt, p_wo_gt


def infer_json_pred(pre_trained_m: PretrainedType4Py, source_file_path: str):
    pre_trained_m.load_pretrained_model()
    src_f_read = read_file(source_file_path)
    src_f_ext = analyze_src_f(src_f_read).to_dict()
    logger.info("Extracted type hints and JSON-representation of input source file")

    logger.info("Predicting type annotations for the given file")
    src_f_ext = infer_single_file(src_f_ext, pre_trained_m)

    save_json(join(dirname(source_file_path),
                              splitext(basename(source_file_path))[0]+"_type4py_typed.json"), src_f_ext)

# def type_check_json_pred(pre_trained_m: PretrainedType4Py, source_file_path: str):
#     pre_trained_m.load_pretrained_model()
#     src_f_read = read_file(source_file_path)
#     src_f_ext = analyze_src_f(src_f_read).to_dict()
#     logger.info("Extracted type hints and JSON-representation of input source file")

#     logger.info("Predicting type annotations for the given file:")
#     src_f_ext = infer_single_file(src_f_ext, pre_trained_m)


def type_check_json_pred(source_file_path: str, tc_resuls: list):

    src_f_read = read_file(source_file_path)
    src_f_ext = load_json(join(dirname(source_file_path),
                              splitext(basename(source_file_path))[0]+"_type4py_typed.json"))
    
    tc_resuls.append((source_file_path, type_check_inferred_types(src_f_ext, src_f_read, join(dirname(source_file_path),
                              splitext(basename(source_file_path))[0]+OUTPUT_FILE_SUFFIX))))


def type_annotate_file(pre_trained_m: PretrainedType4Py, source_code: str, source_file_path: str=None):

    if source_file_path is not None:
        src_f_read = read_file(source_file_path)
    else:
        src_f_read = source_code
    src_f_ext = analyze_src_f(src_f_read).to_dict()
    logger.info("Extracted type hints and JSON-representation of input source file")
    
    src_f_ext = infer_single_file(src_f_ext, pre_trained_m)
    logger.info("Predicted type annotations for the given file")

    # type_check_inferred_types(src_f_ext, src_f_read, join(dirname(source_file_path),
    #                           splitext(basename(source_file_path))[0]+OUTPUT_FILE_SUFFIX))

    # src_f_ext = get_type_checked_preds(src_f_ext, src_f_read, source_file_path)

    return src_f_ext

def predict_types_src_code(pre_trained_m: PretrainedType4Py, src_code: str) -> dict:
    src_f_ext = analyze_src_f(src_code).to_dict()
    logger.info("Extracted type hints and JSON-representation of input source file")

    logger.info("Predicting type annotations for the given file:")
    src_f_ext = infer_single_file(src_f_ext, pre_trained_m)

    return src_f_ext

def infer_main(pre_trained_model_path: str, source_file_path: str):
    
    logger.info(f"Inferring types for the file '{basename(source_file_path)}'' using the Type4Py pretrained model")
    logger.info(f"*************************************************************************")

    f_type_slots_preds = get_type_slots_preds_file(source_file_path)
    print(f_type_slots_preds[1])
    is_tc, p_type = type_check_pred(src_f_read, src_f_o_path, src_f_ext, mypy_tc, p, m_v_t)

    # pre_trained_m = PretrainedType4Py(pre_trained_model_path)
    # pre_trained_m.load_pretrained_model()

    # infer_json_pred(pre_trained_m, source_file_path)

    # src_f_ext = type_annotate_file(pre_trained_m, source_file_path)
    # save_json(join(pre_trained_model_path, splitext(basename(source_file_path))[0]+"_typed.json"), src_f_ext)

