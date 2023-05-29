import os
from typing import List
import pandas as pd
import tqdm
import json

from type4py.deploy.infer import PretrainedType4Py, type_annotate_file
from type4py import logger
from libsa4py.exceptions import ParseError

from libsa4py.utils import list_files, find_repos_list
from pathlib import Path

def find_test_list(project_dir, dataset_split):
    if os.path.exists(dataset_split):
        repos_list: List[dict] = []

        df = pd.read_csv(dataset_split)
        test_df = df[df['set'] == 'test']
        for index, row in test_df.iterrows():
            project = row['project']
            author = project.split('/')[1]
            repo = project.split('/')[2]
            project_path = os.path.join(project_dir, author, repo)
            if os.path.isdir(project_path):
                repos_list.append({"author": author, "repo": repo})
        return repos_list

    else:
        # logger.info(f"dataset_split file: {dataset_split} does not exist!")
        raise FileNotFoundError(f"dataset_split file: {dataset_split} does not exist!")

def infer(repo, model, project_dir, tar_dir):
    project_author = repo["author"]
    project_name = repo["repo"]
    project_path = os.path.join(project_dir, project_author, project_name)
    id_tuple = (project_author, project_name)
    project_id = "/".join(id_tuple)
    project_analyzed_files: dict = {project_id: {"src_files": {}, "type_annot_cove": 0.0}}
    print(f'Running pipeline for project {project_path}')

    print(f'Extracting for {project_path}...')
    project_files = list_files(project_path)
    print(f"{project_path} has {len(project_files)} files")

    project_files = [(f, str(Path(f).relative_to(Path(project_path).parent))) for f in project_files]

    if len(project_files) != 0:
        for filename, f_relative in project_files:
            try:
                ext_type_hints = type_annotate_file(model, None, filename)
                project_analyzed_files[project_id]["src_files"][filename] = \
                    ext_type_hints
            except ParseError as err:
                print("project: %s |file: %s |Exception: %s" % (project_id, filename, err))
            except UnicodeDecodeError:
                print(f"Could not read file {filename}")
            except Exception as err:
                print("project: %s |file: %s |Exception: %s" % (project_id, filename, err))

    if len(project_analyzed_files[project_id]["src_files"].keys()) != 0:
        project_analyzed_files[project_id]["type_annot_cove"] = \
            round(sum([project_analyzed_files[project_id]["src_files"][s]["type_annot_cove"] for s in
                       project_analyzed_files[project_id]["src_files"].keys()]) / len(
                project_analyzed_files[project_id]["src_files"].keys()), 2)

    processed_file = os.path.join(tar_dir, f"{project_author}{project_name}_mlInfer.json")
    with open(processed_file, 'w') as json_f:
        json.dump(project_analyzed_files, json_f, indent=4)


def infer_projects(model, project_dir, tar_dir, split_file):
    if split_file is not None:
        repo_infos_test = find_test_list(project_dir, split_file)
        logger.info(f'Totally find {len(repo_infos_test)} projects in test set')
    else:
        logger.info(f"dataset_split file not provided, infer all projects in {project_dir}")
        repo_infos_test = find_repos_list(project_dir)
        logger.info(f'Totally find {len(repo_infos_test)} projects in project dir')

    for repo in tqdm(repo_infos_test):
        infer(repo, model, project_dir, tar_dir)


def infer_project_main(model_path, input_path, output_path, split_file):
    t4py_pretrained_m = PretrainedType4Py(model_path, "gpu", pre_read_type_cluster=False, use_pca=True)
    t4py_pretrained_m.load_pretrained_model()

    infer_projects(t4py_pretrained_m, input_path, output_path, split_file)

