'''
This module includes function for rebuild project for pyre infer, clean & mask type annotations
'''

from libsa4py.cst_visitor import Visitor
from libsa4py.cst_transformers import TypeAdder, SpaceAdder, StringRemover, CommentAndDocStringRemover, NumberRemover, \
    TypeAnnotationRemover, TypeQualifierResolver
from libsa4py.exceptions import ParseError
from libsa4py.utils import read_file, write_file, find_repos_list, list_files
from pathlib import Path
import libcst as cst
import os
import shutil

def rebuild(filename: str, project_dir: str, tar_dict: str,
            program_types: cst.metadata.type_inference_provider.PyreData = None,
            include_seq2seq: bool = True):
    program = read_file(filename)
    try:
        parsed_program = cst.parse_module(program)
    except Exception as e:
        raise ParseError(str(e))

    # Resolves qualified names for a modules' type annotations
    program_tqr = cst.metadata.MetadataWrapper(parsed_program).visit(TypeQualifierResolver())

    v = Visitor()
    if program_types is not None:
        mw = cst.metadata.MetadataWrapper(program_tqr,
                                          cache={cst.metadata.TypeInferenceProvider: program_types})
        mw.visit(v)
    else:
        mw = cst.metadata.MetadataWrapper(program_tqr, cache={cst.metadata.TypeInferenceProvider: {'types': []}})
        mw.visit(v)

    if include_seq2seq:
        v_type = TypeAnnotationRemover()
        v_untyped = parsed_program.visit(v_type)

        relative_path = str(Path(filename).relative_to(Path(project_dir)))
        tar_path = os.path.join(tar_dict, relative_path)

        write_file(tar_path, v_untyped.code)


def rebuild_repo(project_dir, tar_dir, repo_info):
    repo_path = os.path.join(project_dir, repo_info['author'], repo_info['repo'])
    tar_path = os.path.join(tar_dir, repo_info['author'], repo_info['repo'])
    shutil.copytree(repo_path, tar_path)
    for root, dirs, files in os.walk(tar_path):
        for filename in files:
            os.unlink(os.path.join(root, filename))

    source_files = list_files(repo_path)
    for filename in source_files:
        rebuild(filename, project_dir, tar_dir)