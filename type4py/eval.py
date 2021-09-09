from type4py import logger
from libsa4py.utils import load_json
from os.path import join
from sklearn.metrics import classification_report, precision_score
from tqdm import tqdm
from typing import List, Tuple
import pickle
import re
import numpy as np

logger.name = __name__

def eval_type_embed(y_pred: np.array, y_true: np.array, ubiquitous_types: set, common_types: set,
                    top_n: int=10):

    all_ubiq_types = 0
    corr_ubiq_types = 0
    all_common_types = 0
    corr_common_types = 0
    all_rare_types = 0
    corr_rare_types = 0

    # Mask arrays to keep location correct predictions
    corr_ubiq_mask = np.array([False] * len(y_pred), dtype=np.bool)
    corr_common_mask = np.array([False] * len(y_pred), dtype=np.bool)
    corr_rare_mask = np.array([False] * len(y_pred), dtype=np.bool)

    for idx, p in enumerate(y_pred):
        
        if y_true[idx] in ubiquitous_types:
            all_ubiq_types += 1
            if y_true[idx] in p[:top_n]:
                corr_ubiq_types += 1
                corr_ubiq_mask[idx] = True

        elif y_true[idx] in common_types:
            all_common_types += 1
            if y_true[idx] in p[:top_n]:
                corr_common_types += 1
                corr_common_mask[idx] = True
        else:
            all_rare_types += 1
            if y_true[idx] in p[:top_n]:
                corr_rare_types += 1
                corr_rare_mask[idx] = True

    return (corr_ubiq_types + corr_common_types + corr_rare_types) / len(y_pred) * 100.0, corr_ubiq_types / all_ubiq_types * 100.0, \
            corr_common_types / all_common_types * 100.0, corr_rare_types / all_rare_types * 100.0, corr_common_mask, corr_rare_mask

def eval_parametric_match(y_pred: np.array, y_true: np.array, ubiquitous_types: str,
                          common_types: set, label_enc, top_n: int=10):
    """
    Finds correct parametric types in predicted types. That is, List[*] is parametric type.
    Only outermost is considered, which is List in the given example.
    """

    corr_ubiq_types = 0
    all_param_common_types = 0
    corr_param_common_types = 0
    all_param_rare_types = 0
    corr_param_rare_types = 0
    param_type_match = r'(.+)\[(.+)\]'

    def pred_param_types(pred_types: np.array, true_param_type):
        no_match = 0
        for p in label_enc.inverse_transform(pred_types):
            if re.match(param_type_match, p):
                if true_param_type.group(1) == re.match(param_type_match, p).group(1):
                    no_match += 1
                    break
        
        return no_match

    for idx, t in enumerate(tqdm(y_true, total=len(y_true), desc="Calculating parametric match")):
        
        if t in ubiquitous_types:
            # The selected ubiquitous types are not parametric types
            if t in y_pred[idx][:top_n]:
                corr_ubiq_types += 1
        elif t in common_types:
            all_param_common_types += 1
            if t in y_pred[idx][:top_n]:
                corr_param_common_types += 1
            else:
                matched_param_type = re.match(param_type_match, label_enc.inverse_transform([t])[0])
                if matched_param_type:
                    corr_param_common_types += pred_param_types(y_pred[idx], matched_param_type)

        else:
            all_param_rare_types += 1
            if t in y_pred[idx][:top_n]:
                corr_param_rare_types += 1
            else:
                matched_param_type = re.match(param_type_match, label_enc.inverse_transform([t])[0])
                if matched_param_type:
                    corr_param_rare_types += pred_param_types(y_pred[idx], matched_param_type)

    return (corr_ubiq_types + corr_param_common_types + corr_param_rare_types) / len(y_pred) * 100.0, \
            corr_param_common_types / all_param_common_types * 100.0, corr_param_rare_types / all_param_rare_types * 100.0

def eval_pred_dsl(test_pred: List[dict], common_types, tasks: set, top_n=10, mrr_all=False):
    """
    Computes evaluation metrics such as recall, precision and f1-score
    """
    param_type_match = r'(.+)\[(.+)\]'

    def pred_types_fix(y_true: str, y_pred: List[Tuple[str, int]]):
        for i, (p, _) in enumerate(y_pred[:top_n]):
            if p == y_true:
                return p, 1/(i+1)

        return y_pred[0][0], 0.0

    def is_param_correct(true_param_type: str, pred_types: np.array):
        no_match = 0
        r = 0.0
        for i, p in enumerate(pred_types):
            if re.match(param_type_match, p):
                if re.match(param_type_match, true_param_type).group(1) == re.match(param_type_match, p).group(1):
                    no_match += 1
                    r = 1/(i+1)
                    break
            else:
                if re.match(param_type_match, true_param_type).group(1).lower() == p.lower():
                    no_match += 1
                    r = 1/(i+1)
                    break
        
        return no_match, r

    #ubiquitous_types = {'str', 'int', 'list', 'bool', 'float'}
    ubiquitous_types = {'str', 'int', 'list', 'bool', 'float', 'typing.Text', 'typing.List', 'typing.List[typing.Any]', 'typing.list'}
    #common_types = common_types - ubiquitous_types

    all_ubiq_types = 0
    corr_ubiq_types = 0
    all_common_types = 0
    corr_common_types = 0
    all_rare_types = 0
    corr_rare_types = 0

    all_param_common_types = 0
    corr_param_common_types = 0
    all_param_rare_types = 0
    corr_param_rare_types = 0

    mrr = []
    mrr_exact_ubiq = []
    mrr_exact_comm = []
    mrr_exact_rare = []

    mrr_param_ubiq = []
    mrr_param_comm = []
    mrr_param_rare = []
   
    for p in tqdm(test_pred, total=len(test_pred)):

        if p['task'] not in tasks:
            continue

        top_n_pred, r = pred_types_fix(p['original_type'], p['predictions'])
        mrr.append(r)
        
        if p['original_type'] in ubiquitous_types:
            all_ubiq_types += 1
            mrr_exact_ubiq.append(r)
            if p['original_type'] == top_n_pred:
                corr_ubiq_types += 1
        elif p['original_type'] in common_types:
            all_common_types += 1
            mrr_exact_comm.append(r)
            if p['original_type'] == top_n_pred:
                corr_common_types += 1
            elif p['is_parametric']:
                m, pr = is_param_correct(p['original_type'], [i for i, _ in p['predictions'][:top_n]])
                mrr_param_comm.append(pr)
                corr_param_common_types += m
            # else:
            #     mrr_exact_comm.append(r)
        else:
            all_rare_types += 1
            mrr_exact_rare.append(r)
            if p['original_type'] == top_n_pred:
                corr_rare_types += 1
            elif p['is_parametric']:
                m, pr = is_param_correct(p['original_type'], [i for i, _ in p['predictions'][:top_n]])
                mrr_param_rare.append(pr)
                corr_param_rare_types += m
            # else:
            #     mrr_exact_rare.append(r)

    tasks = 'Combined' if tasks == {'Parameter', 'Return', 'Variable'} else list(tasks)[0]
    logger.info(f"Type4Py - {tasks} - Exact match - all: {(corr_ubiq_types + corr_common_types + corr_rare_types) / (all_ubiq_types+all_common_types+all_rare_types) * 100.0:.1f}%")
    logger.info(f"Type4Py - {tasks} - Exact match - ubiquitous: {corr_ubiq_types / all_ubiq_types * 100.0:.1f}%")
    logger.info(f"Type4Py - {tasks} - Exact match - common: {corr_common_types / all_common_types * 100.0:.1f}%")
    logger.info(f"Type4Py - {tasks} - Exact match - rare: {corr_rare_types / all_rare_types * 100.0:.1f}%")

    logger.info(f"Type4Py - {tasks} - Parametric match - all: {(corr_ubiq_types + corr_common_types + corr_rare_types + corr_param_common_types + corr_param_rare_types) / (all_ubiq_types+all_common_types+all_rare_types) * 100.0:.1f}%")
    logger.info(f"Type4Py - {tasks} - Parametric match - common: {(corr_param_common_types + corr_common_types) / all_common_types * 100.0:.1f}%")
    logger.info(f"Type4Py - {tasks} - Parametric match - rare: {(corr_param_rare_types+corr_rare_types) / all_rare_types * 100.0:.1f}%")
    
    logger.info(f"Type4Py - Mean reciprocal rank {np.mean(mrr)*100:.1f}")

    if mrr_all:
        logger.info(f"Type4Py - {tasks} - MRR - Exact match - all: {np.mean(mrr)*100:.1f}")
        logger.info(f"Type4Py - {tasks} - MRR - Exact match - ubiquitous: {np.mean(mrr_exact_ubiq)*100:.1f}")
        logger.info(f"Type4Py - {tasks} - MRR - Exact match - common: {np.mean(mrr_exact_comm)*100:.1f}")
        logger.info(f"Type4Py - {tasks} - MRR - Exact match - rare: {np.mean(mrr_exact_rare)*100:.1f}")
        #print(mrr_param_comm)
        logger.info(f"Type4Py - {tasks} - MRR - Parameteric match - all: {np.mean(mrr_exact_ubiq+mrr_exact_comm+mrr_exact_rare+mrr_param_comm+mrr_param_rare)*100:.1f}")
        logger.info(f"Type4Py - {tasks} - MRR - Parameteric match - common: {np.mean(mrr_param_comm+mrr_exact_comm)*100:.1f}")
        logger.info(f"Type4Py - {tasks} - MRR - Parameteric match - rare: {np.mean(mrr_param_rare+mrr_exact_rare)*100:.1f}")

    return np.mean(mrr)*100

def evaluate(output_path: str, data_name: str, tasks: set, top_n: int=10, mrr_all=False):

    logger.info(f"Evaluating the Type4Py {data_name} model for {tasks} prediction task")
    logger.info(f"*************************************************************************")
    # Loading label encoder andd common types
    test_pred = load_json(join(output_path, f'type4py_{data_name}_test_predictions.json'))
    le_all = pickle.load(open(join(output_path, "label_encoder_all.pkl"), 'rb'))
    common_types = pickle.load(open(join(output_path, "complete_common_types.pkl"), 'rb'))
    common_types = set(le_all.inverse_transform(list(common_types)))
    #ubiquitous_types = {'str', 'int', 'list', 'bool', 'float'}
    #common_types = common_types - ubiquitous_types
    
    eval_pred_dsl(test_pred, common_types, tasks, top_n=top_n, mrr_all=mrr_all)