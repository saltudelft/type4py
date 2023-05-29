import os
from pathlib import Path
import utils.pyre_utils as pyre_util
from utils.utils import rebuild_repo
from libsa4py.utils import list_files, read_file
from libsa4py.exceptions import ParseError
from libsa4py.cst_extractor import Extractor
import shutil


def pyre_start(project_path):
    pyre_util.clean_watchman_config(project_path)
    pyre_util.clean_pyre_config(project_path)
    pyre_util.start_watchman(project_path)
    pyre_util.start_pyre(project_path)


def pyre_infer(repo, project_dir):
    # rebuild for masking original types
    cache_path = "/cache_path"
    os.mkdir(cache_path)
    rebuild_repo(project_dir, cache_path, repo)

    project_author = repo["author"]
    project_name = repo["repo"]
    project_path = os.path.join(cache_path, project_author, project_name)
    id_tuple = (project_author, project_name)
    project_id = "/".join(id_tuple)
    project_analyzed_files: dict = {project_id: {"src_files": {}, "type_annot_cove": 0.0}}

    print(f'Running pyre pipeline for project {project_path}')
    pyre_start(project_path)
    # start pyre infer for project
    print(f'Running pyre infer for project {project_path}')
    pyre_util.pyre_infer(project_path)
    print(f'Extracting for {project_path}...')
    project_files = list_files(project_path)
    print(f"{project_path} has {len(project_files)} files")

    project_files = [(f, str(Path(f).relative_to(Path(project_path).parent))) for f in project_files]

    if len(project_files) != 0:
        print(f'Running pyre query for project {project_path}')
        try:
            for filename, f_relative in project_files:
                pyre_data_file = pyre_util.pyre_query_types(project_path, filename)
                project_analyzed_files[project_id]["src_files"][filename] = \
                    Extractor.extract(read_file(filename), pyre_data_file).to_dict()
        except ParseError as err:
            print("project: %s |file: %s |Exception: %s" % (project_id, filename, err))
        except UnicodeDecodeError:
            print(f"Could not read file {filename}")
        except Exception as err:
            print("project: %s |file: %s |Exception: %s" % (project_id, filename, err))

    print(f'Saving static analysis results for {project_id}...')

    if len(project_analyzed_files[project_id]["src_files"].keys()) != 0:
        project_analyzed_files[project_id]["type_annot_cove"] = \
            round(sum([project_analyzed_files[project_id]["src_files"][s]["type_annot_cove"] for s in
                       project_analyzed_files[project_id]["src_files"].keys()]) / len(
                project_analyzed_files[project_id]["src_files"].keys()), 2)

    pyre_util.watchman_shutdown(project_path)
    pyre_util.pyre_server_shutdown(project_path)
    pyre_util.clean_config(project_path)

    # remove cache projects
    shutil.rmtree(cache_path)

    return project_analyzed_files
