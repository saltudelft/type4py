'''
Including functions for preprocess among different infer results
- type consis
- remove alias
- check type
'''

import regex

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