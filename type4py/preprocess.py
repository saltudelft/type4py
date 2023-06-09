from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from type4py import logger, AVAILABLE_TYPES_NUMBER, MAX_PARAM_TYPE_DEPTH, AVAILABLE_TYPE_APPLY_PROB
from libsa4py.merge import merge_jsons_to_dict, create_dataframe_fns, create_dataframe_vars
from libsa4py.cst_transformers import ParametricTypeDepthReducer
from libsa4py.cst_lenient_parser import lenient_parse_module
from libsa4py.utils import list_files
from typing import Tuple
from ast import literal_eval
from collections import Counter
from os.path import exists, join
from tqdm import tqdm
import regex
import os
import pickle
import random
import pandas as pd
import numpy as np

logger.name = __name__
tqdm.pandas()

# Precompile often used regex
first_cap_regex = regex.compile('(.)([A-Z][a-z]+)')
all_cap_regex = regex.compile('([a-z0-9])([A-Z])')
sub_regex = r'typing\.|typing_extensions\.|t\.|builtins\.|collections\.'


def make_types_consistent(df_all: pd.DataFrame, df_vars: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes typing module from type annotations
    """

    def remove_quote_types(t: str):
        s = regex.search(r'^\'(.+)\'$', t)
        if bool(s):
            return s.group(1)
        else:
            #print(t)
            return t
    
    df_all['return_type'] = df_all['return_type'].progress_apply(lambda x: regex.sub(sub_regex, "", str(x)) if x else x)
    df_all['arg_types'] = df_all['arg_types'].progress_apply(lambda x: str([regex.sub(sub_regex, "", t) \
                                                       if t else t for t in literal_eval(x)]))
    df_all['return_type'] = df_all['return_type'].progress_apply(remove_quote_types)
    df_all['arg_types'] = df_all['arg_types'].progress_apply(lambda x: str([remove_quote_types(t) if t else t for t in literal_eval(x)]))

    df_vars['var_type'] = df_vars['var_type'].progress_apply(lambda x: regex.sub(sub_regex, "", str(x)))
    df_vars['var_type'] = df_vars['var_type'].progress_apply(remove_quote_types)
    
    return df_all, df_vars

def resolve_type_aliasing(df_param: pd.DataFrame, df_ret: pd.DataFrame,
                          df_vars: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Resolves type aliasing and mappings. e.g. `[]` -> `list`
    """
    import libcst as cst
    # Problematic patterns: (?<=.*)Tuple\[Any, *?.*?\](?<=.*)
    type_aliases = {'(?<=.*)any(?<=.*)|(?<=.*)unknown(?<=.*)': 'Any',
                    '^{}$|^Dict$|^Dict\[\]$|(?<=.*)Dict\[Any, *?Any\](?=.*)|^Dict\[unknown, *Any\]$': 'dict',
                    '^Set$|(?<=.*)Set\[\](?<=.*)|^Set\[Any\]$': 'set',
                    '^Tuple$|(?<=.*)Tuple\[\](?<=.*)|^Tuple\[Any\]$|(?<=.*)Tuple\[Any, *?\.\.\.\](?=.*)|^Tuple\[unknown, *?unknown\]$|^Tuple\[unknown, *?Any\]$|(?<=.*)tuple\[\](?<=.*)': 'tuple',
                    '^Tuple\[(.+), *?\.\.\.\]$': r'Tuple[\1]',
                    '\\bText\\b': 'str',
                    '^\[\]$|(?<=.*)List\[\](?<=.*)|^List\[Any\]$|^List$': 'list',
                    '^\[{}\]$': 'List[dict]',
                    '(?<=.*)Literal\[\'.*?\'\](?=.*)': 'Literal',
                    '(?<=.*)Literal\[\d+\](?=.*)': 'Literal', # Maybe int?!
                    '^Callable\[\.\.\., *?Any\]$|^Callable\[\[Any\], *?Any\]$|^Callable[[Named(x, Any)], Any]$': 'Callable',
                    '^Iterator[Any]$': 'Iterator',
                    '^OrderedDict[Any, *?Any]$': 'OrderedDict',
                    '^Counter[Any]$': 'Counter',
                    '(?<=.*)Match[Any](?<=.*)': 'Match'}

    def resolve_type_alias(t: str):
        for t_alias in type_aliases:
            if regex.search(regex.compile(t_alias), t):
                t = regex.sub(regex.compile(t_alias), type_aliases[t_alias], t)
        return t

    df_param['arg_type'] = df_param['arg_type'].progress_apply(resolve_type_alias)
    df_ret['return_type'] = df_ret['return_type'].progress_apply(resolve_type_alias)
    df_vars['var_type'] = df_vars['var_type'].progress_apply(resolve_type_alias)

    return df_param, df_ret, df_vars

def preprocess_parametric_types(df_param: pd.DataFrame, df_ret: pd.DataFrame,
                                df_vars: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reduces the depth of parametric types
    """
    from libcst import parse_module, ParserSyntaxError
    global s
    s = 0
    def reduce_depth_param_type(t: str) -> str:
        global s
        if regex.match(r'.+\[.+\]', t):
            try:
                t = parse_module(t)
                t = t.visit(ParametricTypeDepthReducer(max_annot_depth=MAX_PARAM_TYPE_DEPTH))
                return t.code
            except ParserSyntaxError:
                try:
                    t = lenient_parse_module(t)
                    t = t.visit(ParametricTypeDepthReducer(max_annot_depth=MAX_PARAM_TYPE_DEPTH))
                    s += 1
                    return t.code
                except ParserSyntaxError:
                    return None
        else:
            return t

    df_param['arg_type'] = df_param['arg_type'].progress_apply(reduce_depth_param_type)
    df_ret['return_type'] = df_ret['return_type'].progress_apply(reduce_depth_param_type)
    df_vars['var_type'] = df_vars['var_type'].progress_apply(reduce_depth_param_type)
    logger.info(f"Sucssesfull lenient parsing {s}")

    return df_param, df_ret, df_vars

def filter_functions(df: pd.DataFrame, funcs=['str', 'unicode', 'repr', 'len', 'doc', 'sizeof']) -> pd.DataFrame:
    """
    Filters functions which are not useful.
    :param df: dataframe to use
    :return: filtered dataframe
    """

    df_len = len(df)
    logger.info(f"Functions before dropping on __*__ methods {len(df):,}")
    df = df[~df['name'].isin(funcs)]
    logger.info(f"Functions after dropping on __*__ methods {len(df):,}")
    logger.info(f"Filtered out {df_len - len(df):,} functions.")

    return df

def filter_variables(df_vars: pd.DataFrame, types=['Any', 'None', 'object', 'type', 'Type[Any]',
                                                   'Type[cls]', 'Type[type]', 'Type', 'TypeVar', 'Optional[Any]']):
    """
    Filters out variables with specified types such as Any or None
    """

    df_var_len = len(df_vars)
    logger.info(f"Variables before dropping on {','.join(types)}: {len(df_vars):,}")
    df_vars = df_vars[~df_vars['var_type'].isin(types)]
    logger.info(f"Variables after dropping on {','.join(types)}: {len(df_vars):,}")
    logger.info(f"Filtered out {df_var_len - len(df_vars):,} variables.")

    return df_vars

def filter_var_wo_type(df_vars: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out variables without a type
    """
    df_var_len = len(df_vars)
    logger.info(f"Variables before dropping: {len(df_vars):,}")
    df_vars = df_vars[df_vars['var_type'].notnull()]
    logger.info(f"Variables after dropping dropping: {len(df_vars):,}")
    logger.info(f"Filtered out {df_var_len - len(df_vars):,} variables w/o a type.")

    return df_vars

def gen_argument_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a new dataframe containing all argument data.
    :param df: dataframe for which to extract argument
    :return: argument dataframe
    """
    arguments = []
    for i, row in tqdm(df.iterrows(), total=len(df.index), desc="Processing arguments"):
        for p_i, arg_name in enumerate(literal_eval(row['arg_names'])):

            # Ignore self arg
            if arg_name == 'self':
                continue

            arg_type = literal_eval(row['arg_types'])[p_i].strip('\"')

            # Ignore Any or None types
            # TODO: Ignore also object type
            # TODO: Ignore Optional[Any]
            if arg_type == '' or arg_type in {'Any', 'None', 'object'}:
                continue

            arg_descr = literal_eval(row['arg_descrs'])[p_i]
            arg_occur = [a.replace('self', '').strip() if 'self' in a.split() else a for a in literal_eval(row['args_occur'])[p_i]]
            other_args = " ".join([a for a in literal_eval(row['arg_names']) if a != 'self'])
            arguments.append([row['file'], row['name'], row['func_descr'], arg_name, arg_type, arg_descr, other_args, arg_occur, row['aval_types']])

    return pd.DataFrame(arguments, columns=['file', 'func_name', 'func_descr', 'arg_name', 'arg_type', 'arg_comment', 'other_args',
                                            'arg_occur', 'aval_types'])

def filter_return_dp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters return datapoints based on a set of criteria.
    """

    logger.info(f"Functions before dropping on return type {len(df):,}")
    df = df.dropna(subset=['return_type'])
    logger.info(f"Functions after dropping on return type {len(df):,}")

    logger.info(f"Functions before dropping nan, None, Any return type {len(df):,}")
    to_drop = np.invert((df['return_type'] == 'nan') | (df['return_type'] == 'None') | (df['return_type'] == 'Any'))
    df = df[to_drop]
    logger.info(f"Functions after dropping nan return type {len(df):,}")

    logger.info(f"Functions before dropping on empty return expression {len(df):,}")
    df = df[df['return_expr'].apply(lambda x: len(literal_eval(x))) > 0]
    logger.info(f"Functions after dropping on empty return expression {len(df):,}")

    return df

def format_df(df: pd.DataFrame) -> pd.DataFrame:
    df['arg_names'] = df['arg_names'].apply(lambda x: literal_eval(x))
    df['arg_types'] = df['arg_types'].apply(lambda x: literal_eval(x))
    df['arg_descrs'] = df['arg_descrs'].apply(lambda x: literal_eval(x))
    df['return_expr'] = df['return_expr'].apply(lambda x: literal_eval(x))

    return df

def encode_all_types(df_ret: pd.DataFrame, df_params: pd.DataFrame, df_vars: pd.DataFrame,
                     output_dir: str):
    all_types = np.concatenate((df_ret['return_type'].values, df_params['arg_type'].values,
                                df_vars['var_type'].values), axis=0)
    le_all = LabelEncoder()
    le_all.fit(all_types)
    df_ret['return_type_enc_all'] = le_all.transform(df_ret['return_type'].values)
    df_params['arg_type_enc_all'] = le_all.transform(df_params['arg_type'].values)
    df_vars['var_type_enc_all'] = le_all.transform(df_vars['var_type'].values)

    unq_types, count_unq_types = np.unique(all_types, return_counts=True)
    pd.DataFrame(
            list(zip(le_all.transform(unq_types), [unq_types[i] for i in np.argsort(count_unq_types)[::-1]],
                    [count_unq_types[i] for i in np.argsort(count_unq_types)[::-1]])),
            columns=['enc', 'type', 'count']
        ).to_csv(os.path.join(output_dir, "_most_frequent_all_types.csv"), index=False)

    logger.info(f"Total no. of extracted types: {len(all_types):,}")
    logger.info(f"Total no. of unique types: {len(unq_types):,}")

    return df_ret, df_params, le_all

def gen_most_frequent_avl_types(avl_types_dir, output_dir, top_n: int = 1024) -> pd.DataFrame:
    """
    It generates top n most frequent available types
    :param top_n:
    :return:
    """

    aval_types_files = [os.path.join(avl_types_dir, f) for f in os.listdir(avl_types_dir) if os.path.isfile(os.path.join(avl_types_dir, f))]

    # All available types across all Python projects
    all_aval_types = []

    for f in aval_types_files:
        with open(f, 'r') as f_aval_type:
            all_aval_types = all_aval_types + f_aval_type.read().splitlines()

    counter = Counter(all_aval_types)

    df = pd.DataFrame.from_records(counter.most_common(top_n), columns=['Types', 'Count'])
    df.to_csv(os.path.join(output_dir, "top_%d_types.csv" % top_n), index=False)

    return df

def encode_aval_types(df_param: pd.DataFrame, df_ret: pd.DataFrame, df_var: pd.DataFrame,
                      df_aval_types: pd.DataFrame, apply_rvth: bool=False):
    """
    It encodes the type of parameters and return according to visible type hints.
    apply_rvth: bool, In production, there are no visible type hints (VTHs) available as the user's code may not have type annotations.
                      Therefore, the model needs to learn from a training set where there are not VTHs available at least half of the time.
    """

    types = df_aval_types['Types'].tolist()

    def trans_aval_type(x):
        for i, t in enumerate(types):
            if x in t:
                if not apply_rvth or random.random() > AVAILABLE_TYPE_APPLY_PROB:
                    return i
        return len(types) - 1

    if not apply_rvth:
        # If the arg type doesn't exist in top_n available types, we insert n + 1 into the vector as it represents the other type.
        df_param['param_aval_enc'] = df_param['arg_type'].progress_apply(trans_aval_type)
        df_ret['ret_aval_enc'] = df_ret['return_type'].progress_apply(trans_aval_type)
        df_var['var_aval_enc'] = df_var['var_type'].progress_apply(trans_aval_type)
    else:
        logger.info(f"Encoding available type hints with a probability of {AVAILABLE_TYPE_APPLY_PROB:.2f}")
        df_param['param_aval_enc'] = df_param['aval_types'].progress_apply(lambda x: trans_aval_type(x))
        df_ret['ret_aval_enc'] = df_ret['aval_types'].progress_apply(lambda x: trans_aval_type(x))
        df_var['var_aval_enc'] = df_var['aval_types'].progress_apply(lambda x: trans_aval_type(x))

    return df_param, df_ret

def preprocess_ext_fns(output_dir: str, limit: int = None, apply_random_vth: bool = False):
    """
    Applies preprocessing steps to the extracted functions
    """

    if not (os.path.exists(os.path.join(output_dir, "all_fns.csv")) and os.path.exists(os.path.join(output_dir, "all_vars.csv"))):
        logger.info("Merging JSON projects")
        if os.path.exists(os.path.join(output_dir, 'processed_projects')):
            merged_jsons = merge_jsons_to_dict(list_files(os.path.join(output_dir, 'processed_projects'), ".json"), limit)
        elif os.path.exists(os.path.join(output_dir, 'processed_projects_complete')):
            merged_jsons = merge_jsons_to_dict(list_files(os.path.join(output_dir, 'processed_projects_complete'), ".json"), limit)
        else:
            raise RuntimeError("Could not find processed projects in the ManyTypes4Py dataset")
        logger.info("Creating functions' Dataframe")
        create_dataframe_fns(output_dir, merged_jsons)
        logger.info("Creating variables' Dataframe")
        create_dataframe_vars(output_dir, merged_jsons)
        
    logger.info("Loading vars & fns Dataframe")
    processed_proj_fns = pd.read_csv(os.path.join(output_dir, "all_fns.csv"), low_memory=False)
    processed_proj_vars = pd.read_csv(os.path.join(output_dir, "all_vars.csv"), low_memory=False)

    # Split the processed files into train, validation and test sets
    if all(processed_proj_fns['set'].isin(['train', 'valid', 'test'])) and \
       all(processed_proj_vars['set'].isin(['train', 'valid', 'test'])):
        logger.info("Found the sets split in the input dataset")
        train_files = processed_proj_fns['file'][processed_proj_fns['set'] == 'train']
        valid_files = processed_proj_fns['file'][processed_proj_fns['set'] == 'valid']
        test_files = processed_proj_fns['file'][processed_proj_fns['set'] == 'test']

        train_files_vars = processed_proj_vars['file'][processed_proj_vars['set'] == 'train']
        valid_files_vars = processed_proj_vars['file'][processed_proj_vars['set'] == 'valid']
        test_files_vars = processed_proj_vars['file'][processed_proj_vars['set'] == 'test']

    else:
        logger.info("Splitting sets randomly")
        uniq_files = np.unique(np.concatenate((processed_proj_fns['file'].to_numpy(), processed_proj_vars['file'].to_numpy())))
        train_files, test_files = train_test_split(pd.DataFrame(uniq_files, columns=['file']), test_size=0.2)
        train_files, valid_files = train_test_split(pd.DataFrame(train_files, columns=['file']), test_size=0.1)
        train_files_vars, valid_files_vars, test_files_vars = train_files, valid_files, test_files

    df_train = processed_proj_fns[processed_proj_fns['file'].isin(train_files.to_numpy().flatten())]
    logger.info(f"No. of functions in train set: {df_train.shape[0]:,}")
    df_valid = processed_proj_fns[processed_proj_fns['file'].isin(valid_files.to_numpy().flatten())]
    logger.info(f"No. of functions in validation set: {df_valid.shape[0]:,}")
    df_test = processed_proj_fns[processed_proj_fns['file'].isin(test_files.to_numpy().flatten())]
    logger.info(f"No. of functions in test set: {df_test.shape[0]:,}")

    df_var_train = processed_proj_vars[processed_proj_vars['file'].isin(train_files_vars.to_numpy().flatten())]
    logger.info(f"No. of variables in train set: {df_var_train.shape[0]:,}")
    df_var_valid = processed_proj_vars[processed_proj_vars['file'].isin(valid_files_vars.to_numpy().flatten())]
    logger.info(f"No. of variables in validation set: {df_var_valid.shape[0]:,}")
    df_var_test = processed_proj_vars[processed_proj_vars['file'].isin(test_files_vars.to_numpy().flatten())]
    logger.info(f"No. of variables in test set: {df_var_test.shape[0]:,}")

    assert list(set(df_train['file'].tolist()).intersection(set(df_test['file'].tolist()))) == []
    assert list(set(df_train['file'].tolist()).intersection(set(df_valid['file'].tolist()))) == []
    assert list(set(df_test['file'].tolist()).intersection(set(df_valid['file'].tolist()))) == []

    # Exclude variables without a type
    processed_proj_vars = filter_var_wo_type(processed_proj_vars)

    logger.info(f"Making type annotations consistent")
    # Makes type annotations consistent by removing `typing.`, `t.`, and `builtins` from a type.
    processed_proj_fns, processed_proj_vars = make_types_consistent(processed_proj_fns, processed_proj_vars)

    assert any([bool(regex.match(sub_regex, str(t))) for t in processed_proj_fns['return_type']]) == False
    assert any([bool(regex.match(sub_regex, t)) for t in processed_proj_fns['arg_types']]) == False
    assert any([bool(regex.match(sub_regex, t)) for t in processed_proj_vars['var_type']]) == False

    # Filters variables with type Any or None
    processed_proj_vars = filter_variables(processed_proj_vars)

    # Filters trivial functions such as `__str__` and `__len__` 
    processed_proj_fns = filter_functions(processed_proj_fns)
    
    # Extracts type hints for functions' arguments
    processed_proj_fns_params = gen_argument_df(processed_proj_fns)

    # Filters out functions: (1) without a return type (2) with the return type of Any or None (3) without a return expression
    processed_proj_fns = filter_return_dp(processed_proj_fns)
    processed_proj_fns = format_df(processed_proj_fns)

    logger.info(f"Resolving type aliases")
    # Resolves type aliasing and mappings. e.g. `[]` -> `list`
    processed_proj_fns_params, processed_proj_fns, processed_proj_vars = resolve_type_aliasing(processed_proj_fns_params,
                                                                                               processed_proj_fns,
                                                                                               processed_proj_vars)

    assert any([bool(regex.match(r'^{}$|\bText\b|^\[{}\]$|^\[\]$', t)) for t in processed_proj_fns['return_type']]) == False
    assert any([bool(regex.match(r'^{}$|\bText\b|^\[\]$', t)) for t in processed_proj_fns_params['arg_type']]) == False

    logger.info(f"Preproceessing parametric types")
    processed_proj_fns_params, processed_proj_fns, processed_proj_vars = preprocess_parametric_types(processed_proj_fns_params,
                                                                                                     processed_proj_fns,
                                                                                                     processed_proj_vars)
    # Exclude variables without a type
    processed_proj_vars = filter_var_wo_type(processed_proj_vars)

    processed_proj_fns, processed_proj_fns_params, le_all = encode_all_types(processed_proj_fns, processed_proj_fns_params,
                                                                             processed_proj_vars, output_dir)

    # Exclude self from arg names and return expressions
    processed_proj_fns['arg_names_str'] = processed_proj_fns['arg_names'].apply(lambda l: " ".join([v for v in l if v != 'self']))
    processed_proj_fns['return_expr_str'] = processed_proj_fns['return_expr'].apply(lambda l: " ".join([regex.sub(r"self\.?", '', v) for v in l]))

    # Drop all columns useless for the ML model
    processed_proj_fns = processed_proj_fns.drop(columns=['author', 'repo', 'has_type', 'arg_names', 'arg_types', 'arg_descrs', 'args_occur',
                         'return_expr'])

    # Visible type hints
    if exists(join(output_dir, 'MT4Py_VTHs.csv')):
        logger.info("Using visible type hints")
        processed_proj_fns_params, processed_proj_fns = encode_aval_types(processed_proj_fns_params, processed_proj_fns,
                                                                          processed_proj_vars,
                                                                          pd.read_csv(join(output_dir, 'MT4Py_VTHs.csv')).head(AVAILABLE_TYPES_NUMBER),
                                                                          apply_random_vth)
    else:
        logger.info("Using naive available type hints")
        df_types = gen_most_frequent_avl_types(os.path.join(output_dir, "extracted_visible_types"), output_dir, AVAILABLE_TYPES_NUMBER)
        processed_proj_fns_params, processed_proj_fns = encode_aval_types(processed_proj_fns_params, processed_proj_fns,
                                                                        processed_proj_vars, df_types, apply_random_vth)

    # Split parameters and returns type dataset by file into a train and test sets
    df_params_train = processed_proj_fns_params[processed_proj_fns_params['file'].isin(train_files.to_numpy().flatten())]
    df_params_valid = processed_proj_fns_params[processed_proj_fns_params['file'].isin(valid_files.to_numpy().flatten())]
    df_params_test = processed_proj_fns_params[processed_proj_fns_params['file'].isin(test_files.to_numpy().flatten())]

    df_ret_train = processed_proj_fns[processed_proj_fns['file'].isin(train_files.to_numpy().flatten())]
    df_ret_valid = processed_proj_fns[processed_proj_fns['file'].isin(valid_files.to_numpy().flatten())]
    df_ret_test = processed_proj_fns[processed_proj_fns['file'].isin(test_files.to_numpy().flatten())]

    df_var_train = processed_proj_vars[processed_proj_vars['file'].isin(train_files_vars.to_numpy().flatten())]
    df_var_valid = processed_proj_vars[processed_proj_vars['file'].isin(valid_files_vars.to_numpy().flatten())]
    df_var_test = processed_proj_vars[processed_proj_vars['file'].isin(test_files_vars.to_numpy().flatten())]


    assert list(set(df_params_train['file'].tolist()).intersection(set(df_params_test['file'].tolist()))) == []
    assert list(set(df_params_train['file'].tolist()).intersection(set(df_params_valid['file'].tolist()))) == []
    assert list(set(df_params_test['file'].tolist()).intersection(set(df_params_valid['file'].tolist()))) == []

    assert list(set(df_ret_train['file'].tolist()).intersection(set(df_ret_test['file'].tolist()))) == []
    assert list(set(df_ret_train['file'].tolist()).intersection(set(df_ret_valid['file'].tolist()))) == []
    assert list(set(df_ret_test['file'].tolist()).intersection(set(df_ret_valid['file'].tolist()))) == []

    assert list(set(df_var_train['file'].tolist()).intersection(set(df_var_test['file'].tolist()))) == []
    assert list(set(df_var_train['file'].tolist()).intersection(set(df_var_valid['file'].tolist()))) == []
    assert list(set(df_var_test['file'].tolist()).intersection(set(df_var_valid['file'].tolist()))) == []

    # Store the dataframes and the label encoders
    logger.info("Saving preprocessed functions on the disk...")
    with open(os.path.join(output_dir, "label_encoder_all.pkl"), 'wb') as file:
        pickle.dump(le_all, file)
    
    df_params_train.to_csv(os.path.join(output_dir, "_ml_param_train.csv"), index=False)
    df_params_valid.to_csv(os.path.join(output_dir, "_ml_param_valid.csv"), index=False)
    df_params_test.to_csv(os.path.join(output_dir, "_ml_param_test.csv"), index=False)

    df_ret_train.to_csv(os.path.join(output_dir, "_ml_ret_train.csv"), index=False)
    df_ret_valid.to_csv(os.path.join(output_dir, "_ml_ret_valid.csv"), index=False)
    df_ret_test.to_csv(os.path.join(output_dir, "_ml_ret_test.csv"), index=False)

    df_var_train.to_csv(os.path.join(output_dir, "_ml_var_train.csv"), index=False)
    df_var_valid.to_csv(os.path.join(output_dir, "_ml_var_valid.csv"), index=False)
    df_var_test.to_csv(os.path.join(output_dir, "_ml_var_test.csv"), index=False)