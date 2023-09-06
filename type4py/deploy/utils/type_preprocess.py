'''
Including functions for preprocess among different infer results
- type consis
- remove alias
- check type
'''

import regex
from libsa4py.nl_preprocessing import NLPreprocessor


def apply_nlp_transf(extracted_module):
    """
    Applies NLP transformation to identifiers in a module
    """
    nlp_prep = NLPreprocessor()

    def fn_nlp_transf(fn_d, nlp_prep: NLPreprocessor):
        fn_d['name'] = nlp_prep.process_identifier(fn_d['name'])
        fn_d['params'] = {nlp_prep.process_identifier(p): t for p, t in fn_d['params'].items()}
        fn_d['ret_exprs'] = [nlp_prep.process_identifier(r.replace('return ', '')) for r in fn_d['ret_exprs']]
        fn_d['params_occur'] = {p: [nlp_prep.process_sentence(j) for i in o for j in i] for p, o in
                                fn_d['params_occur'].items()}
        fn_d['variables'] = {nlp_prep.process_identifier(v): t for v, t in fn_d['variables'].items()}
        fn_d['fn_var_occur'] = {v: [nlp_prep.process_sentence(j) for i in o for j in i] for v, o in
                                fn_d['fn_var_occur'].items()}
        fn_d['params_descr'] = {nlp_prep.process_identifier(p): nlp_prep.process_sentence(fn_d['params_descr'][p]) \
                                for p in fn_d['params_descr'].keys()}
        fn_d['docstring']['func'] = nlp_prep.process_sentence(fn_d['docstring']['func'])
        fn_d['docstring']['ret'] = nlp_prep.process_sentence(fn_d['docstring']['ret'])
        fn_d['docstring']['long_descr'] = nlp_prep.process_sentence(fn_d['docstring']['long_descr'])
        return fn_d

    extracted_module['variables'] = {nlp_prep.process_identifier(v): t for v, t in
                                     extracted_module['variables'].items()}
    extracted_module['mod_var_occur'] = {v: [nlp_prep.process_sentence(j) for i in o for j in i] for v,o in extracted_module['mod_var_occur'].items()}

    for c in extracted_module['classes']:
        c['variables'] = {nlp_prep.process_identifier(v): t for v, t in c['variables'].items()}
        c['cls_var_occur'] = {v: [nlp_prep.process_sentence(j) for i in o for j in i] for v, o in
                              c['cls_var_occur'].items()}
        c['funcs'] = [fn_nlp_transf(f, nlp_prep) for f in c['funcs']]
    extracted_module['funcs'] = [fn_nlp_transf(f, nlp_prep) for f in extracted_module['funcs']]

    return extracted_module

def check(t: str):
    types = ["", "Any", "any", "None", "Object", "object", "type", "Type[Any]",
             'Type[cls]', 'Type[type]', 'Type', 'TypeVar', 'Optional[Any]']
    if t in types:
        return False
    else:
        return True

def make_types_consistent(t: str):
    """
    Removes typing module from type annotations
    """
    sub_regex = r'typing\.|typing_extensions\.|t\.|builtins\.|collections\.'

    def remove_quote_types(t: str):
        s = regex.search(r'^\'(.+)\'$', t)
        if bool(s):
            return s.group(1)
        else:
            # print(t)
            return t

    t = regex.sub(sub_regex, "", str(t))
    t = remove_quote_types(t)
    return t


def resolve_type_aliasing(t: str):
    """
    Resolves type aliasing and mappings. e.g. `[]` -> `list`
    """
    type_aliases = {'(?<=.*)any(?<=.*)|(?<=.*)unknown(?<=.*)': 'Any',
                    '^{}$|^Dict$|^Dict\[\]$|(?<=.*)Dict\[Any, *?Any\](?=.*)|^Dict\[unknown, *Any\]$': 'dict',
                    '^Set$|(?<=.*)Set\[\](?<=.*)|^Set\[Any\]$': 'set',
                    '^Tuple$|(?<=.*)Tuple\[\](?<=.*)|^Tuple\[Any\]$|(?<=.*)Tuple\[Any, *?\.\.\.\](?=.*)|^Tuple\[unknown, *?unknown\]$|^Tuple\[unknown, *?Any\]$|(?<=.*)tuple\[\](?<=.*)': 'tuple',
                    '^Tuple\[(.+), *?\.\.\.\]$': r'Tuple[\1]',
                    '\\bText\\b': 'str',
                    '^\[\]$|(?<=.*)List\[\](?<=.*)|^List\[Any\]$|^List$': 'list',
                    '^\[{}\]$': 'List[dict]',
                    '(?<=.*)Literal\[\'.*?\'\](?=.*)': 'Literal',
                    '(?<=.*)Literal\[\d+\](?=.*)': 'Literal',  # Maybe int?!
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

    return resolve_type_alias(t)