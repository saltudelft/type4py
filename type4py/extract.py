from typing import Tuple, Type, Union, List, Optional, Any, Dict, Pattern, Match
from astor import code_gen
from tokenize import tokenize
from io import BytesIO
import math
import ast
import re
import asttokens
import docstring_parser
import pandas as pd


def window_tokens(tokens, str, win_size):
    """
    Extracts a window of size n around a string.
    """

    R = int(math.floor(win_size / 2))
    L = int(math.floor(win_size / 2) - 1) if win_size % 2 == 0 else int(math.floor(win_size / 2))

    str_loc = tokens.index(str)

    num_el_L = (str_loc - 1) + 1
    num_el_R = (len(tokens) - 1) - str_loc

    left_part = []
    right_part = []

    if L <= num_el_L:
        left_part = [tokens[str_loc - i] for i in range(1, L+1)]
    elif num_el_L >= 2:
        left_part = [tokens[i] for i in range(num_el_L)]
    elif num_el_L == 1:
        left_part = [tokens[0]]

    if R <= num_el_R:
        right_part = [tokens[i] for i in range(str_loc + 1, str_loc + 1 + R)]
    elif num_el_R >= 2:
        right_part = tokens[str_loc+1:num_el_R+1]
        right_part = [tokens[i] for i in range(str_loc + 1, str_loc + 1 + num_el_R)]
    elif num_el_R == 1:
        right_part = [tokens[str_loc+1]]

    return left_part + [tokens[str_loc]] + right_part

class Function:
    """
    Representation of a parsed python function
    """

    def __init__(self, name: str, docstring: str, func_descr: Optional[str], arg_names: List[str], arg_types: List[str],
                 arg_descrs: Optional[List[str]], args_occur, return_type: str, return_expr: List[str],
                 return_descr: Optional[str], variables: List[str], variables_types: List[str]) -> None:
        self.name = name
        self.docstring = docstring
        self.func_descr = func_descr
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.arg_descrs = arg_descrs
        self.return_type = return_type
        self.return_expr = return_expr  # Return expressions
        self.return_descr = return_descr

        # Occurence of arguments
        self.args_occur = args_occur

        # Additional variables
        self.variables = variables
        self.variables_types = variables_types

    def __eq__(self, other) -> bool:
        if isinstance(other, Function):
            return self.name == other.name and \
                   self.docstring == other.docstring and \
                   self.arg_names == other.arg_names and \
                   self.arg_types == other.arg_types and \
                   self.return_type == other.return_type and \
                   self.return_expr == other.return_expr and \
                   self.func_descr == other.func_descr and \
                   self.arg_descrs == other.arg_descrs and \
                   self.return_descr == other.return_descr and \
                   self.variables == other.variables and \
                   self.variables_types == other.variables_types

        return False

    def __repr__(self) -> str:
        values = list(map(lambda x: repr(x), self.__dict__.values()))
        values = ",".join(values)
        return "Function(%s)" % values

    def has_types(self) -> bool:
        return any(ty for ty in self.arg_types) or self.return_type != ''

    def as_tuple(self) -> Tuple:
        return tuple(self.__dict__.values())

    def tuple_keys(self) -> Tuple:
        return tuple(self.__dict__.keys())


class Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        # fn here is tuple: (node, [return_node1, ...])
        self.fns = []
        self.return_exprs = []

        self.class_defs = []  # Names of classes
        self.imports = []
    def visit_FunctionDef(self, node) -> None:
        """
        When visiting (async) function definitions,
        save the main function node as well as all its return expressions
        """
        old_return_exprs = self.return_exprs
        self.return_exprs = []

        self.generic_visit(node)

        self.fns.append((node, self.return_exprs))
        self.return_exprs = old_return_exprs

    # Visiting async function is the same as sync
    # for the purpose of extracting names, types etc
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Return(self, node) -> None:
        self.return_exprs.append(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """
        Extracts locally defined types.
        :return:
        """
        self.class_defs.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        """
        Extracts imported types.
        """
        self.imports.append(node.names[0].name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """
        Extracts imported types.
        """
        for n in node.names:
            self.imports.append(n.name)
        self.generic_visit(node)


class ParseError(Exception):
    def __init__(self, msg: str):
        super().__init__("ParseError: " + msg)

class Extractor():
    """
    Extract data from python source code    `
    """

    # Only (async) function definitions contain a return value
    types_to_extract: Tuple[Type[ast.AsyncFunctionDef], Type[ast.FunctionDef]] \
        = (ast.AsyncFunctionDef, ast.FunctionDef)

    ast_fn_def: Any = Union[ast.AsyncFunctionDef, ast.FunctionDef]

    def extract_fn(self, node: ast_fn_def, return_exprs: List[ast.Return]) \
            -> Function:

        function_name: str = self.extract_name(node)
        #print("Function: ", function_name)
        docstring: str = self.extract_docstring(node)
        (arg_names, arg_types) = self.extract_args(node)
        return_type: str = self.extract_return_type(node)
        exprs: List[str] = [self.pretty_print(re) for re in return_exprs]

        docstring_descr: Dict[
            str, Union[Union[Optional[str], Dict[str, str]], Optional[str]]] = self.extract_docstring_descriptions(
            self.check_docstring(docstring))
        arg_descrs: List[str] = list(map(
            lambda arg_name: docstring_descr["params"][arg_name] if arg_name in docstring_descr['params'] else '',
            arg_names
        ))

        #print(arg_descrs)

        #print("arg_names", arg_names)
        args_occur = self.extract_args_ret_occur(node.body, arg_names, 5)
        #print("Args occur: ", args_occur)
        #print("###############################################################################")

        f: Function = Function(function_name, docstring, docstring_descr["function_descr"],
                               arg_names, arg_types, arg_descrs, args_occur,
                               return_type, exprs, docstring_descr["return_descr"], [], [])

        return f

    def extract_name(self, node: ast_fn_def) -> str:
        """Extract the name of the function"""
        return node.name

    def extract_docstring(self, node: ast_fn_def) -> str:
        """Extract the docstring from a function"""
        return ast.get_docstring(node) or ""

    def extract_docstring_descriptions(self, docstring: str) -> Dict[
        str, Union[Union[Optional[str], Dict[str, str]], Optional[str]]]:
        """Extract the return description from the docstring"""
        try:
            parsed_docstring: docstring_parser.parser.Docstring = docstring_parser.parse(docstring)
        except Exception:
            return {"function_descr": None, "params": {}, "return_descr": None}

        descr_map: Dict[str, Union[Union[str, Dict[str, str]], Optional[str]]] = {
            "function_descr": parsed_docstring.short_description,
            "params": {},
            "return_descr": None}

        if parsed_docstring.returns is not None:
            descr_map["return_descr"] = parsed_docstring.returns.description

        for param in parsed_docstring.params:
            descr_map["params"][param.arg_name] = param.description

        return descr_map

    def check_docstring(self, docstring: str) -> str:
        """Check the docstring if it has a valid structure for parsing and returns a valid docstring."""
        dash_line_matcher: Pattern[str] = re.compile("\s*--+")
        param_keywords: List[str] = ["Parameters", "Params", "Arguments", "Args"]
        return_keywords: List[str] = ["Returns", "Return"]
        break_keywords: List[str] = ["See Also", "Examples"]

        convert_docstring: bool = False
        add_indent: bool = False
        add_double_colon: bool = False
        active_keyword: bool = False
        end_docstring: bool = False

        preparsed_docstring: str = ""
        lines: List[str] = docstring.split("\n")
        for line in lines:
            result: Optional[Match] = re.match(dash_line_matcher, line)
            if result is not None:
                preparsed_docstring = preparsed_docstring[:-1] + ":" + "\n"
                convert_docstring = True
            else:
                for keyword in param_keywords:
                    if keyword in line:
                        add_indent = True
                        active_keyword = True
                        break
                if not active_keyword:
                    for keyword in return_keywords:
                        if keyword in line:
                            add_indent = True
                            add_double_colon = True
                            active_keyword = True
                            break
                if not add_double_colon:
                    for keyword in break_keywords:
                        if keyword in line:
                            end_docstring = True
                            break
                if end_docstring:
                    break
                if active_keyword:
                    preparsed_docstring += line + "\n"
                    active_keyword = False
                elif add_double_colon:
                    preparsed_docstring += "\t" + line + ":\n"
                    add_double_colon = False
                elif add_indent:
                    line_parts = line.split(":")
                    if len(line_parts) > 1:
                        preparsed_docstring += "\t" + line_parts[0] + "(" + line_parts[1].replace(" ", "") + "):\n"
                    else:
                        preparsed_docstring += "\t" + line + "\n"
                else:
                    preparsed_docstring += line + "\n"

        if convert_docstring:
            return preparsed_docstring
        else:
            return docstring

    def pretty_print(self, node: Optional[ast.AST]) -> str:
        """
        Given some AST type expression,
        pretty print the type so that it is readable
        """
        if node is None:
            return ""
        return code_gen.to_source(node, indent_with="").rstrip()

    def extract_args(self, node: ast_fn_def) -> Tuple[List[str], List[str]]:
        """Extract the names and types of the function args"""
        arg_names: List[str] = [arg.arg for arg in node.args.args]
        arg_types: List[str] = [self.pretty_print(arg.annotation)
                                for arg in node.args.args]

        if node.args.vararg is not None:
            arg_names.append(node.args.vararg.arg)
            arg_types.append(self.pretty_print(node.args.vararg.annotation))

        return (arg_names, arg_types)

    def extract_return_type(self, node: ast_fn_def) -> str:
        """Extract the return type of the function"""
        return self.pretty_print(node.returns)

    def extract_args_ret_occur(self, func_body, arg_names, win_size):
        """
        Extracts code occurrences of the to-be-typed program element.
        """
        args_occur = []

        for arg in arg_names:
            occur = ""
            for s in self.tokenize_func_stats(func_body):
                if arg in s and 'return' not in s:
                    # Considering a window of tokens around each occurence
                    if len(s) > win_size:
                        occur = occur + " ".join(window_tokens(s, arg, win_size))
                    else:
                        occur = occur + " ".join(s)

            if occur != "":
                args_occur.append(occur)
        
        return args_occur
  
    def tokenize_func_stats(self, func):
        """
        A helper method to tokenize every statement in a method
        :return:
        """
        stats_tokens = []
        ignore_toks = ['utf-8', '', ',']

        for c in func:

            line = self.asttok.get_text(c).encode("utf-8")
            stats_tokens.append([tokval.rstrip() for _, tokval, _, _, _ in tokenize(BytesIO(line).readline) if tokval.rstrip() not in ignore_toks])

        return stats_tokens
    
    def extract(self, program: str) -> Tuple[List[Function], list]:
        """Extract useful data from python program"""
        try:
            self.asttok = asttokens.ASTTokens(program, parse=True)
        except Exception as err:
            raise ParseError(err)

        v: Visitor = Visitor()
        v.visit(self.asttok.tree)

        return list(map(lambda fn: self.extract_fn(*fn), v.fns)), v.class_defs + v.imports

def parse_df(files: list, batch_size=10) -> pd.DataFrame:
    """
    Given a set of files, load each as a dataframe and concatenate the result
    into a large file dataframe containing the data from all files.
    """
    df_merged = None
    batch = []
    total = 0
    for i, file in enumerate(files):
        if i % 100 == 0:
            print(f"Loaded {i}/{len(files)} files")

        _loaded = pd.read_csv(file, low_memory=False)
        total += len(_loaded)
        if df_merged is None:
            df_merged = _loaded
        else:
            batch.append(_loaded)

        if i % batch_size == 0 and i > 0:
            frames = [df_merged, *batch]
            df_merged = pd.concat(frames, ignore_index=True)
            batch = []

    if len(batch) > 0:
        df_merged = pd.concat([df_merged, *batch])

    df_merged = df_merged.reset_index(drop=True)
    assert total == len(df_merged), "Length of all loaded should be equal to merged"

    return df_merged


