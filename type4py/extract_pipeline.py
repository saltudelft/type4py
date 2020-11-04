import os
import pandas as pd
from joblib import delayed
from type4py.utils import ParallelExecutor, filter_directory, list_files, read_file, mk_dir_not_exist, find_repos_list
from type4py.preprocess import NLPreprocessor
from type4py.extract import Extractor, ParseError, parse_df
import traceback

class Pipeline:

    def __init__(self, repos_dir, output_dir):
        self.repos_dir = repos_dir
        self.output_dir = output_dir
        self.avl_types_dir = None
        self.processed_projects_path = None
        self.extractor = Extractor()

        self.__make_output_dirs()

    def __make_output_dirs(self):
        mk_dir_not_exist(self.output_dir)

        self.processed_projects_path = os.path.join(self.output_dir, "processed_projects")
        self.avl_types_dir = os.path.join(self.output_dir, "ext_visible_types")

        mk_dir_not_exist(self.processed_projects_path)
        mk_dir_not_exist(self.avl_types_dir)

    def get_project_filename(self, project) -> str:
        """
        Return the filename at which a project datafile should be stored.
        :param project: the project dict
        :return: return filename
        """
        return os.path.join(self.processed_projects_path, f"{project['author']}{project['repo']}-functions.csv")

    def write_project(self, project) -> None:
        functions = []
        columns = None

        if 'files' in project:
            for file in project['files']:
                for funcs in file['functions']:
                    if columns is None:
                        columns = ['author', 'repo', 'file', 'has_type'] + list(funcs.tuple_keys()) + ['aval_types']

                    function_metadata = (
                                            project['author'],
                                            project['repo'],
                                            file['filename'],
                                            funcs.has_types()
                                        ) + funcs.as_tuple() + (file['aval_types'],)

                    functions.append(function_metadata)

                    assert len(function_metadata) == len(columns), \
                        f"Assertion failed size of columns should be same as the size of the data tuple."

        if len(functions) == 0:
            print("Skipped...")
            return
        function_df = pd.DataFrame(functions, columns=columns)
        function_df['arg_names_len'] = function_df['arg_names'].apply(len)
        function_df['arg_types_len'] = function_df['arg_types'].apply(len)
        function_df.to_csv(self.get_project_filename(project), index=False)

    def merge_processed_projects(self):
        processed_proj_f = list_files(self.processed_projects_path)
        print("Found %d datafiles" % len(processed_proj_f))
        df = parse_df(processed_proj_f, batch_size=4098)
        print("Dataframe loaded writing it to CSV")
        df.to_csv(os.path.join(self.output_dir, '_all_data.csv'), index=False)

    def run(self, jobs: int, no_proj_limit: int=None):
        """
        Run the pipeline (clone, filter, extract, remove) for all given projects
        """

        repos_list = find_repos_list(self.repos_dir) if no_proj_limit is None else find_repos_list(self.repos_dir)[:no_proj_limit]
        ParallelExecutor(n_jobs=jobs)(total=len(repos_list))(
            delayed(self.process_project)(i, project) for i, project in enumerate(repos_list))
        
        self.merge_processed_projects()

    def process_project(self, i, project):
        try:
            project_id = f'{project["author"]}/{project["repo"]}'
            print(f'Running pipeline for project {i} {project_id}')

            # if os.path.exists(self.get_project_filename(project)) and USE_CACHE:
            #     print(f"Found cached copy for project {project_id}")
            #     return

            project['files'] = []

            print(f'Filtering for {project_id}...')
            filtered_project_directory = filter_directory(os.path.join(self.repos_dir, project["author"], project["repo"]))

            print(f'Extracting for {project_id}...')
            extracted_functions = {}
            extracted_avl_types = []
            for filename in list_files(filtered_project_directory):
                try:
                    functions, avl_types = self.extractor.extract(read_file(filename))
                    extracted_functions[filename] = (functions, avl_types)
                    extracted_avl_types = extracted_avl_types + avl_types
                except ParseError:
                    print(f"Could not parse file {filename}")
                except UnicodeDecodeError:
                    print(f"Could not read file {filename}")
                except:
                    # Other unexpected exceptions; Failure of single file should not
                    # fail the entire project processing.
                    # TODO: A better workaround would be to have a specialized exception thrown
                    # by the extractor, so that this exception is specialized.
                    print(f"Could not process file {filename}")

            print(f'Preprocessing for {project_id}...')
            preprocessed_functions = {}
            for filename, ext_funcs_aval_types in extracted_functions.items():
                preprocessed_functions[filename] = ([NLPreprocessor.preprocess(function) for function in ext_funcs_aval_types[0]],
                                                    [NLPreprocessor.process_sentence(aval_t) for aval_t in ext_funcs_aval_types[1]])

            project['files'] = [{'filename': filename, 'functions': ext_funcs_aval_types[0], 'aval_types': list(filter(None, ext_funcs_aval_types[1]))}
                                for filename, ext_funcs_aval_types in preprocessed_functions.items()]

            #print("Available types: ", extracted_avl_types)
            if extracted_avl_types:
                with open(os.path.join(self.avl_types_dir, f'{project["author"]}_{project["repo"]}_avltypes.txt'), 'w') as f:
                    for t in extracted_avl_types:
                        f.write("%s\n" % t)

        except KeyboardInterrupt:
            quit(1)
        except Exception:
            print(f'Running pipeline for project {i} failed')
            traceback.print_exc()
        finally:
            self.write_project(project)