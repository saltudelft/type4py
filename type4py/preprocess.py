from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from type4py import logger
from libsa4py.merge import merge_jsons_to_dict, create_dataframe_fns
from libsa4py.utils import list_files
from ast import literal_eval
from collections import Counter
from tqdm import tqdm
import re
import os
import pickle
import pandas as pd
import numpy as np

logger.name = __name__

# Precompile often used regex
first_cap_regex = re.compile('(.)([A-Z][a-z]+)')
all_cap_regex = re.compile('([a-z0-9])([A-Z])')

# class NLPreprocessor:
#     @staticmethod
#     def preprocess(function: Function) -> Function:
#         """
#         Preprocess a function's comments and identifiers by removing punctuating, removing stopwords and lemmatization
#         """
#
#         return Function(
#             name=NLPreprocessor.process_identifier(function.name),
#             docstring=NLPreprocessor.process_sentence(function.docstring),
#             func_descr=NLPreprocessor.process_sentence(function.func_descr),
#             arg_names=[NLPreprocessor.process_identifier(arg_name) for arg_name in function.arg_names],
#             arg_types=function.arg_types,
#             arg_descrs=[NLPreprocessor.process_sentence(arg_descr) for arg_descr in function.arg_descrs],
#             args_occur=[NLPreprocessor.process_sentence(args_occur) for args_occur in function.args_occur],
#             return_type=function.return_type,
#             return_expr=[NLPreprocessor.process_identifier(expr.replace('return ', '')) for expr in function.return_expr],
#             return_descr=NLPreprocessor.process_sentence(function.return_descr),
#             variables=[NLPreprocessor.process_identifier(var_name) for var_name in function.variables],
#             variables_types=function.variables_types
#         )
#
#     @staticmethod
#     def process_sentence(sentence: str) -> Optional[str]:
#         """
#         Process a natural language sentence
#         """
#
#         if sentence is None:
#             return None
#
#         pipeline = [
#             SentenceProcessor.replace_digits_with_space,
#             SentenceProcessor.remove_punctuation_and_linebreaks,
#             SentenceProcessor.tokenize,
#             SentenceProcessor.lemmatize,
#             SentenceProcessor.remove_stop_words
#         ]
#
#         return reduce(lambda s, action: action(s), pipeline, sentence)
#
#     @staticmethod
#     def process_identifier(sentence: str) -> str:
#         """
#         Process a sentence mainly consisting of identifiers
#
#         Similar to process_sentence, but does not remove stop words.
#         """
#         pipeline = [
#             SentenceProcessor.replace_digits_with_space,
#             SentenceProcessor.remove_punctuation_and_linebreaks,
#             SentenceProcessor.tokenize,
#             SentenceProcessor.lemmatize
#         ]
#
#         return reduce(lambda s, action: action(s), pipeline, sentence)


# class SentenceProcessor:
#     """
#     A collection of static functions to process a natural language sentence
#     """
#
#     @staticmethod
#     def process_sentence(sentence: str) -> Optional[str]:
#         """
#         Process a natural language sentence
#         """
#
#         if sentence is None:
#             return None
#
#         pipeline = [
#             SentenceProcessor.replace_digits_with_space,
#             SentenceProcessor.remove_punctuation_and_linebreaks,
#             SentenceProcessor.tokenize,
#             SentenceProcessor.lemmatize,
#             SentenceProcessor.remove_stop_words
#         ]
#
#         return reduce(lambda s, action: action(s), pipeline, sentence)
#
#     @staticmethod
#     def replace_digits_with_space(sentence: str) -> str:
#         """
#         Replaces digits with a space
#         """
#         return re.sub('[0-9]+', ' ', sentence)
#
#     @staticmethod
#     def remove_punctuation_and_linebreaks(sentence: str) -> str:
#         """
#         Removes and replaces non-textual elements
#
#         Removes whitespace and all punctuations. Question marks and full stops are replaced with
#         a space. Full stops that are not followed by a space are also replaced with a space, e.g. object.property ->
#         object property.
#         """
#         return re.sub('[^A-Za-z0-9 ]+', ' ', sentence) \
#             .replace('\n', '') \
#             .replace('\r', '')
#
#     @staticmethod
#     def tokenize(sentence: str) -> str:
#         """
#         Tokenize camel case and snake case in a sentence and convert the sentence to lower case
#         """
#         sentence = sentence.replace("_", " ")
#         sentence = SentenceProcessor.convert_camelcase(sentence)
#
#         return sentence.lower()
#
#     @staticmethod
#     def lemmatize(sentence: str) -> str:
#         """
#         Lemmatize a sentence (e.g. running -> run)
#         """
#         words = [word for word in sentence.split(' ') if word != '']
#
#         lemmatized = []
#         for token, tag in nltk.pos_tag(words):
#             word_pos = SentenceProcessor.get_wordnet_pos(tag)
#             lemmatizer = nltk.WordNetLemmatizer()
#             try:
#                 if word_pos != '':
#                     lemmatized.append(lemmatizer.lemmatize(token, pos=word_pos))
#                 else:
#                     lemmatized.append(lemmatizer.lemmatize(token))
#             except UnicodeDecodeError:
#                 logger.error(f'Lemmatization failed for {token}, tag: {tag}, word pos: {word_pos}')
#
#         return ' '.join(lemmatized)
#
#     @staticmethod
#     def remove_stop_words(sentence: str) -> str:
#         """
#         Remove stop words from a sentence
#         """
#         return ' '.join([word for word in sentence.split(' ') if word not in nltk.corpus.stopwords.words('english')])
#
#     @staticmethod
#     def get_wordnet_pos(treebank_tag: str) -> str:
#         """
#         Get the WordNet part-of-speech constant for the treebank tag
#         """
#         if treebank_tag.startswith('J'):
#             return nltk.corpus.wordnet.ADJ
#         elif treebank_tag.startswith('V'):
#             return nltk.corpus.wordnet.VERB
#         elif treebank_tag.startswith('N'):
#             return nltk.corpus.wordnet.NOUN
#         elif treebank_tag.startswith('R'):
#             return nltk.corpus.wordnet.ADV
#         else:
#             return ''
#
#     @staticmethod
#     def convert_camelcase(sentence: str) -> str:
#         """
#         Convert `camelCase` to `camel case`.
#         """
#         words = [all_cap_regex.sub(r'\1 \2', first_cap_regex.sub(r'\1 \2', word)) for word in sentence.split(" ")]
#
#         return ' '.join(words)

def make_types_consistent(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Removes typing module from type annotations
    """
    
    df_all['return_type'] = df_all['return_type'].apply(lambda x: re.sub(r'typing\.|t\.|builtins\.', "", str(x)) if x else x)
    df_all['arg_types'] = df_all['arg_types'].apply(lambda x: str([re.sub(r'typing\.|t\.|builtins\.', "", t) \
                                                       if t else t for t in literal_eval(x)]))
    
    return df_all

def resolve_type_aliasing(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Resolves type aliasing and mappings. e.g. `[]` -> `list`
    """

    def resolve_alias(alias_dict: dict, t: str):
        for t_alias in alias_dict:
            if re.search(re.compile(t_alias), t):
                return re.sub(re.compile(t_alias), alias_dict[t_alias], t)
        return None

    def resolve_type_alias_params(types_list):
        type_alias_params = {'^{}$': 'dict', '\\bText\\b': 'str', '^\[\]$': 'list'}
        params_types = []
        for t in literal_eval(types_list):
            resolved_alias = resolve_alias(type_alias_params, t)
            if resolved_alias:
                params_types.append(resolved_alias)
            else:
                params_types.append(t)
            
        return str(params_types)

    def resolve_type_alias_ret(ret_type):
        type_alias_ret = {'^{}$': 'dict', '\\bText\\b': 'str', '^\[{}\]$': 'List[dict]',
                          '^\[\]$': 'list'}
        if ret_type:
            resolved_alias = resolve_alias(type_alias_ret, str(ret_type))
            if resolved_alias:
                return resolved_alias
        
        return ret_type

    df_all['return_type'] = df_all['return_type'].apply(resolve_type_alias_ret)
    df_all['arg_types'] = df_all['arg_types'].apply(resolve_type_alias_params)

    return df_all

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
            if arg_type == '' or arg_type == 'Any' or arg_type == 'None':
                continue

            arg_descr = literal_eval(row['arg_descrs'])[p_i]
            arg_occur = [a.replace('self', '').strip() if 'self' in a.split() else a for a in literal_eval(row['args_occur'])[p_i]]
            other_args = " ".join([a for a in literal_eval(row['arg_names']) if a != 'self'])
            arguments.append([row['file'], row['name'], row['func_descr'], arg_name, arg_type, arg_descr, other_args, arg_occur])

    return pd.DataFrame(arguments, columns=['file', 'func_name', 'func_descr', 'arg_name', 'arg_type', 'arg_comment', 'other_args',
                                            'arg_occur'])

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

def encode_all_types(df_ret: pd.DataFrame, df_params: pd.DataFrame,
                     output_dir: str):
    all_types = np.concatenate((df_ret['return_type'].values, df_params['arg_type'].values), axis=0)
    le_all = LabelEncoder()
    le_all.fit(all_types)
    df_ret['return_type_enc_all'] = le_all.transform(df_ret['return_type'].values)
    df_params['arg_type_enc_all'] = le_all.transform(df_params['arg_type'].values)

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

def encode_aval_types(df_param: pd.DataFrame, df_ret: pd.DataFrame, df_aval_types: pd.DataFrame):
    """
    It encodes the type of parameters and return according to visible type hints
    """

    types = df_aval_types['Types'].tolist()

    def trans_aval_type(x):
        for i, t in enumerate(types):
            if x in t:
                return i
        return len(types) - 1

    # If the arg type doesn't exist in top_n available types, we insert n + 1 into the vector as it represents the other type.
    df_param['param_aval_enc'] = df_param['arg_type'].apply(trans_aval_type)
    df_ret['ret_aval_enc'] = df_ret['return_type'].apply(trans_aval_type)

    return df_param, df_ret

def preprocess_ext_fns(output_dir: str):
    """
    Applies preprocessing steps to the extracted functions
    """

    logger.info("Merging JSON projects and loading functions' Dataframe")
    create_dataframe_fns(output_dir, merge_jsons_to_dict(list_files(os.path.join(output_dir, 'processed_projects'), ".json")))
    processed_proj_fns = pd.read_csv(os.path.join(output_dir, "all_fns.csv"), low_memory=False)

    # Split the processed files into train, validation and test sets
    train_files, test_files = train_test_split(pd.DataFrame(processed_proj_fns['file'].unique(), columns=['file']),
                                               test_size=0.2)
    train_files, valid_files = train_test_split(pd.DataFrame(processed_proj_fns[processed_proj_fns['file'].isin(train_files.to_numpy().flatten())]['file'].unique(),
                                                 columns=['file']), test_size=0.1)

    df_train = processed_proj_fns[processed_proj_fns['file'].isin(train_files.to_numpy().flatten())]
    logger.info(f"No. of functions in train set: {df_train.shape[0]:,}")
    df_valid = processed_proj_fns[processed_proj_fns['file'].isin(valid_files.to_numpy().flatten())]
    logger.info(f"No. of functions in validation set: {df_valid.shape[0]:,}")
    df_test = processed_proj_fns[processed_proj_fns['file'].isin(test_files.to_numpy().flatten())]
    logger.info(f"No. of functions in test set: {df_test.shape[0]:,}")

    assert list(set(df_train['file'].tolist()).intersection(set(df_test['file'].tolist()))) == []
    assert list(set(df_train['file'].tolist()).intersection(set(df_valid['file'].tolist()))) == []
    assert list(set(df_test['file'].tolist()).intersection(set(df_valid['file'].tolist()))) == []

    # Makes type annotations consistent by removing `typing.`, `t.`, and `builtins` from a type.
    processed_proj_fns = make_types_consistent(processed_proj_fns)

    assert any([bool(re.match(r'.*typing\..+|.*t\..+|.*builtins\..+', str(t))) for t in processed_proj_fns['return_type']]) == False
    assert any([bool(re.match(r'.*typing\..+|.*t\..+|.*builtins\..+', t)) for t in processed_proj_fns['arg_types']]) == False

    # Resolves type aliasing and mappings. e.g. `[]` -> `list`
    processed_proj_fns = resolve_type_aliasing(processed_proj_fns)

    assert any([bool(re.match(r'^{}$|\bText\b|^\[{}\]$|^\[\]$', str(t))) for t in processed_proj_fns['return_type']]) == False
    assert any([bool(re.match(r'^{}$|\bText\b|^\[\]$', t)) for type_list in processed_proj_fns['arg_types'] for t in literal_eval(type_list)]) == False

    # Filters trivial functions such as `__str__` and `__len__` 
    processed_proj_fns = filter_functions(processed_proj_fns)
    
    # Extracts type hints for functions' arguments
    processed_proj_fns_params = gen_argument_df(processed_proj_fns)

    # Filters out functions: (1) without a return type (2) with the return type of Any or None (3) without a return expression
    processed_proj_fns = filter_return_dp(processed_proj_fns)

    processed_proj_fns = format_df(processed_proj_fns)

    processed_proj_fns, processed_proj_fns_params, le_all = encode_all_types(processed_proj_fns, processed_proj_fns_params, output_dir)

    # Exclude self from arg names and return expressions
    processed_proj_fns['arg_names_str'] = processed_proj_fns['arg_names'].apply(lambda l: " ".join([v for v in l if v != 'self']))
    processed_proj_fns['return_expr_str'] = processed_proj_fns['return_expr'].apply(lambda l: " ".join([re.sub(r"self\.?", '', v) for v in l]))

    # Drop all columns useless for the ML model
    processed_proj_fns = processed_proj_fns.drop(columns=['author', 'repo', 'has_type', 'arg_names', 'arg_types', 'arg_descrs', 'args_occur',
                         'return_expr'])

    # Find most frequent visible type hints
    df_types = gen_most_frequent_avl_types(os.path.join(output_dir, "extracted_visible_types"), output_dir)
    processed_proj_fns_params, processed_proj_fns = encode_aval_types(processed_proj_fns_params, processed_proj_fns, df_types)

    # Split parameters and returns type dataset by file into a train and test sets
    df_params_train = processed_proj_fns_params[processed_proj_fns_params['file'].isin(train_files.to_numpy().flatten())]
    df_params_valid = processed_proj_fns_params[processed_proj_fns_params['file'].isin(valid_files.to_numpy().flatten())]
    df_params_test = processed_proj_fns_params[processed_proj_fns_params['file'].isin(test_files.to_numpy().flatten())]

    df_ret_train = processed_proj_fns[processed_proj_fns['file'].isin(train_files.to_numpy().flatten())]
    df_ret_valid = processed_proj_fns[processed_proj_fns['file'].isin(valid_files.to_numpy().flatten())]
    df_ret_test = processed_proj_fns[processed_proj_fns['file'].isin(test_files.to_numpy().flatten())]

    assert list(set(df_params_train['file'].tolist()).intersection(set(df_params_test['file'].tolist()))) == []
    assert list(set(df_params_train['file'].tolist()).intersection(set(df_params_valid['file'].tolist()))) == []
    assert list(set(df_params_test['file'].tolist()).intersection(set(df_params_valid['file'].tolist()))) == []

    assert list(set(df_ret_train['file'].tolist()).intersection(set(df_ret_test['file'].tolist()))) == []
    assert list(set(df_ret_train['file'].tolist()).intersection(set(df_ret_valid['file'].tolist()))) == []
    assert list(set(df_ret_test['file'].tolist()).intersection(set(df_ret_valid['file'].tolist()))) == []

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