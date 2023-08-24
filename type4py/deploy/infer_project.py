"""
This module is for infer projects and output json files based on three approaches:
type4py, pyre and pyright
"""
import os
from typing import List
import pandas as pd
import tqdm

from type4py.deploy.infer import PretrainedType4Py, type_annotate_file
from type4py.deploy.utils.extract_types import extract_result_ml
from type4py import logger
from libsa4py.exceptions import ParseError

from libsa4py.utils import list_files, find_repos_list, save_json, load_json
from pathlib import Path
import multiprocessing
from type4py.deploy.static_infer import pyre_infer, pyright_infer
from type4py.deploy.utils.pyre_merge import merge_pyre
from type4py.deploy.utils.pyright_merge import merge_pyright


ml_queue = multiprocessing.Queue()
pyre_queue = multiprocessing.Queue()
pyright_queue = multiprocessing.Queue()

def find_test_list(project_dir, dataset_split):
    if os.path.exists(dataset_split):
        repos_list: List[dict] = []

        test_df = pd.read_csv(dataset_split)
        for index, row in test_df.iterrows():
            project = row['project']
            author = project.split('/')[0]
            repo = project.split('/')[1]
            project_path = os.path.join(project_dir, author, repo)
            if os.path.isdir(project_path):
                repos_list.append({"author": author, "repo": repo})
        return repos_list

    else:
        print("test_repo.csv does not exist!")

def ml_infer(repo, model, project_dir):
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

    return project_analyzed_files


def run_mlInfer(repo, model, project_dir):
    ml_result = ml_infer(repo, model, project_dir)
    ml_queue.put(ml_result)

def run_pyreInfer(repo, project_dir):
    pyre_result = pyre_infer(repo, project_dir)
    pyre_queue.put(pyre_result)

def run_pyrightInfer():
    pyright_result = pyright_infer(repo, project_dir)
    pyright_queue.put(pyright_result)

def infer_projects(model, project_dir, tar_dir, approach, split_file):
    if split_file is not None:
        repo_infos_test = find_test_list(project_dir, split_file)
        logger.info(f'Totally find {len(repo_infos_test)} projects in test set')
    else:
        logger.info(f"dataset_split file not provided, infer all projects in {project_dir}")
        repo_infos_test = find_repos_list(project_dir)
        logger.info(f'Totally find {len(repo_infos_test)} projects in project dir')

    if approach == "t4py":
        predict_list = []
        os.makedirs("ml_res", exist_ok=True)
        ml_dir = "ml_res"
        for repo in repo_infos_test:
            project_name = "".join((repo["author"], repo["repo"]))
            project_id = "/".join((repo["author"], repo["repo"]))
            filepath = os.path.join(ml_dir, f"{project_name}_mlInfer.json")
            processed_file = ml_infer(repo, model, project_dir)
            save_json(filepath, processed_file)
            label_filename = "".join((repo["author"], repo["repo"])) + ".json"
            label_file = load_json(os.path.join(tar_dir, "processed_projects", label_filename))
            ml_predicts = extract_result_ml(label_file, processed_file, project_id)
            predict_list.extend(ml_predicts)
        save_json(os.path.join(tar_dir, f"{approach}_complete_test_predictions.json"),predict_list)

    if approach == "t4pyre":
        predict_list = []
        os.makedirs("t4pyre_res", exist_ok=True)
        ml_dir = "t4pyre_res"
        for repo in repo_infos_test:
            process1 = multiprocessing.Process(target=run_mlInfer, args=(repo, model, project_dir))
            process2 = multiprocessing.Process(target=run_pyreInfer, args = (repo, project_dir))

            # Start the processes
            process1.start()
            process2.start()

            # Get the results from t4py and pyre & merge
            ml_result = ml_queue.get()
            sa_result = pyre_queue.get()

            project_id = "/".join((repo["author"], repo["repo"]))
            project_name = "".join((repo["author"], repo["repo"]))
            hy_result = merge_pyre(ml_result, sa_result, project_id)

            filepath = os.path.join(ml_dir, f"{project_name}_t4pyreInfer.json")
            save_json(filepath, hy_result)
            label_filename = "".join((repo["author"], repo["repo"])) + ".json"
            label_file = load_json(os.path.join(tar_dir, "processed_projects", label_filename))
            t4pyre_predicts = extract_result_ml(label_file, hy_result, project_id)
            predict_list.extend(t4pyre_predicts)

        save_json(os.path.join(tar_dir, f"{approach}_complete_test_predictions.json"), predict_list)


    if approach == "t4pyright":
        for repo in tqdm(repo_infos_test):
            process1 = multiprocessing.Process(target=run_mlInfer)
            process2 = multiprocessing.Process(target=run_pyrightInfer)

            # Start the processes
            process1.start()
            process2.start()

            # Get the results from t4py and pyright & merge
            ml_result = ml_queue.get()
            sa_result = pyright_queue.get()

            project_id = "/".join((repo["author"], repo["repo"]))
            project_name = "".join((repo["author"], repo["repo"]))
            hy_result = merge_pyright(ml_result, sa_result, project_id)

            filepath = os.path.join(tar_dir, f"{project_name}_t4pyrightInfer.json")
            save_json(filepath, hy_result)

def infer_project_main(model_path, input_path, output_path, approach, split_file):
    t4py_pretrained_m = PretrainedType4Py(model_path, "gpu", pre_read_type_cluster=False, use_pca=True)
    t4py_pretrained_m.load_pretrained_model()
    infer_projects(t4py_pretrained_m, input_path, output_path, approach, split_file)

