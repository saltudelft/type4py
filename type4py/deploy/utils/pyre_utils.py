"""
Helper functions to use pyre in the pipeline
"""
from pathlib import Path
from subprocess import TimeoutExpired
from os.path import join, exists, isdir
from libcst.metadata.type_inference_provider import PyreData
import os
import shutil
import json
import subprocess
from typing import List, Optional, Tuple



def run_command(cmd_args: List[str], timeout: Optional[int] = None
                ) -> Tuple[str, str, int]:
    process = subprocess.run(cmd_args, shell=True, capture_output=True, timeout=timeout)
    return process.stdout.decode(), process.stderr.decode(), process.returncode


def clean_watchman_config(project_path: str):
    # update the watchman config to the project
    dict = {"root": "."}
    if exists(join(project_path, '.watchmanconfig')):
        os.remove(join(project_path, '.watchmanconfig'))
        print(f"[WATCHMAN_CLEAN] config of {project_path} ")

    with open(join(project_path, '.watchmanconfig'), "w") as f:
        json.dump(dict, f)
        print(f"[WATCHMAN_WRITE] config of {project_path} ")


def clean_pyre_config(project_path: str):
    # update pyre config file for the path
    dict = {
        "site_package_search_strategy": "pep561",
        "source_directories": [
            "."
        ],
        "typeshed": "/pyre-check/stubs/typeshed/typeshed",
        "workers":64
    }

    if exists(join(project_path, '.pyre_configuration')):
        os.remove(join(project_path, '.pyre_configuration'))
        print(f"[PYRE_CLEAN] config of {project_path} ")

    with open(join(project_path, '.pyre_configuration'), "w") as f:
        json.dump(dict, f)
        print(f"[PYRE_WRITE] config of {project_path} ")


def start_watchman(project_path: str):
    # start watchman server
    stdout, stderr, r_code = run_command(
        "cd %s; watchman watch-project ." % project_path)
    if r_code == 0:
        print(f"[WATCHMAN SERVER] started at {project_path} ", stdout, stderr)
    else:
        print(f"[WATCHMAN_ERROR] p: {project_path}", stderr)


def start_pyre(project_path: str):
    # start pyre server
    stdout, stderr, r_code = run_command(
        "cd %s; pyre start" % project_path)
    print(f"[PYRE_SERVER] started at {project_path} ", stdout, stderr)


def pyre_infer(project_path: str):
    # start pyre server for the project
    stdout, stderr, r_code = run_command(
        "cd %s; pyre infer; pyre infer -i --annotate-from-existing-stubs" % project_path)
    print(f"[PYRE_INFER] started at {project_path} ", stdout, stderr)


def pyre_query_types(project_path: str, file_path: str, timeout: int = 600) -> Optional[PyreData]:
    try:
        file_types = None
        stdout, stderr, r_code = run_command('''cd %s; pyre query "types(path='%s')"''' % (project_path,
                                                                                           str(Path(
                                                                                               file_path).relative_to(
                                                                                               Path(project_path)))),
                                             timeout=timeout)
        if r_code == 0:
            file_types = json.loads(stdout)["response"][0]
        else:
            print(f"[PYRE_ERROR] p: {project_path}", stderr)
    except KeyError:
        print(f"[PYRE_ERROR] p: {project_path}", json.loads(stdout)['error'])
    except TimeoutExpired as te:
        print(f"[PYRE_TIMEOUT] p: {project_path}", te)
    finally:
        return file_types


def pyre_server_shutdown(project_path: str):
    # stop pyre server in the project path
    stdout, stderr, r_code = run_command("cd %s ; pyre stop" % project_path)
    print(f"[PYRE_SERVER] stopped at {project_path} ", stdout, stderr)

def watchman_shutdown(project_path: str):
    # stop pyre server in the project path
    stdout, stderr, r_code = run_command("cd %s ; watchman watch-del ." % project_path)
    print(f"[WATCHMAN SERVER] stopped at {project_path} ", stdout, stderr)


def clean_config(project_path: str):
    # clean watchman
    if exists(join(project_path, '.watchmanconfig')):
        os.remove(join(project_path, '.watchmanconfig'))
        print(f"[WATCHMAN_CLEAN] config of {project_path} ")

    # clean pyre
    if exists(join(project_path, '.pyre_configuration')):
        os.remove(join(project_path, '.pyre_configuration'))
        print(f"[PYRE_CLEAN] config of {project_path} ")

    # clean pyre folder
    pyre_dir = join(project_path, '.pyre')
    if exists(pyre_dir) and isdir(pyre_dir):
        shutil.rmtree(pyre_dir)

