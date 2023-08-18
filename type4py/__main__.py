from type4py import data_loaders
from type4py.to_onnx import type4py_to_onnx
from type4py.reduce import reduce_tc
from type4py.utils import setup_logs_file
from type4py.exceptions import InferApproachNotFound
from libsa4py.cst_pipeline import Pipeline
from libsa4py.utils import find_repos_list
import argparse
import warnings

warnings.filterwarnings("ignore")

data_loading_comb = {'train': data_loaders.load_combined_train_data, 'valid': data_loaders.load_combined_valid_data,
                     'test': data_loaders.load_combined_test_data, 'labels': data_loaders.load_combined_labels,
                     'name': 'complete'}

# add split data loading function
data_loading_comb_sep = {'train': data_loaders.load_combined_train_data_split,
                         'valid': data_loaders.load_combined_valid_data_split,
                         'test': data_loaders.load_combined_test_data,
                         'labels': data_loaders.load_combined_labels_split,
                         'name': 'complete'}

data_loading_woi = {'train': data_loaders.load_combined_train_data_woi,
                    'valid': data_loaders.load_combined_valid_data_woi,
                    'test': data_loaders.load_combined_test_data_woi, 'labels': data_loaders.load_combined_labels,
                    'name': 'woi'}

data_loading_woc = {'train': data_loaders.load_combined_train_data_woc,
                    'valid': data_loaders.load_combined_valid_data_woc,
                    'test': data_loaders.load_combined_test_data_woc, 'labels': data_loaders.load_combined_labels,
                    'name': 'woc'}

data_loading_wov = {'train': data_loaders.load_combined_train_data_wov,
                    'valid': data_loaders.load_combined_valid_data_wov,
                    'test': data_loaders.load_combined_test_data_wov, 'labels': data_loaders.load_combined_labels,
                    'name': 'wov'}

data_loading_param = {'train': data_loaders.load_param_train_data, 'valid': data_loaders.load_param_valid_data,
                      'test': data_loaders.load_param_test_data, 'labels': data_loaders.load_param_labels,
                      'name': 'argument'}

data_loading_ret = {'train': data_loaders.load_ret_train_data, 'valid': data_loaders.load_ret_valid_data,
                    'test': data_loaders.load_ret_test_data, 'labels': data_loaders.load_ret_labels,
                    'name': 'return'}

data_loading_var = {'train': data_loaders.load_var_train_data, 'valid': data_loaders.load_var_valid_data,
                    'test': data_loaders.load_var_test_data, 'labels': data_loaders.load_var_labels,
                    'name': 'variable'}


def extract(args):
    p = Pipeline(args.c, args.o, True, False, args.d)
    p.run(find_repos_list(args.c) if args.l is None else find_repos_list(args.c)[:args.l], args.w)

def preprocess(args):
    from type4py.preprocess import preprocess_ext_fns
    setup_logs_file(args.o, "preprocess")
    preprocess_ext_fns(args.o, args.l, args.rvth)

def vectorize(args):
    from type4py.vectorize import vectorize_args_ret
    setup_logs_file(args.o, "vectorize")
    vectorize_args_ret(args.o)

def learn(args):
    from type4py.learn import train
    setup_logs_file(args.o, "learn")
    if args.woi:
        train(args.o, data_loading_woi, args.p, args.v)
    elif args.woc:
        train(args.o, data_loading_woc, args.p, args.v)
    elif args.wov:
        train(args.o, data_loading_wov, args.p, args.v)
    else:
        train(args.o, data_loading_comb, args.p, args.v)

# add learn_split function for CLI command "learn_split"
def learn_split(args):
    from type4py.learn_split import train_split
    setup_logs_file(args.o, "learn_sep")
    if args.c:
        train_split(args.o, data_loading_comb_sep, args.dt, args.p, args.v)

def predict(args):
    from type4py.predict import test
    setup_logs_file(args.o, "predict")
    if args.woi:
        test(args.o, data_loading_woi, args.l)
    elif args.woc:
        test(args.o, data_loading_woc, args.l)
    elif args.wov:
        test(args.o, data_loading_wov, args.l)
    elif args.c:
        test(args.o, data_loading_comb, args.l, args.rtc)

# add gen_cluster function for CLI command "gen_clu"
def gen_type_cluster(args):
    from type4py.gen_type_cluster import gen_type_cluster
    setup_logs_file(args.o, "gen_clusters")
    gen_type_cluster(args.o, data_loading_comb_sep, args.dt)

def predict_split(args):
    from type4py.predict_split import test_split
    setup_logs_file(args.o, "predict_sep")
    if args.c:
        test_split(args.o, data_loading_comb)

def eval(args):
    from type4py.eval import evaluate
    setup_logs_file(args.o, "eval")
    tasks = {'c': {'Parameter', 'Return', 'Variable'}, 'p': {'Parameter'},
             'r': {'Return'}, 'v': {'Variable'}}
    if args.woi:
        evaluate(args.o, data_loading_woi['name'], tasks[args.t], args.tp, args.mrr)
    elif args.woc:
        evaluate(args.o, data_loading_woc['name'], tasks[args.t], args.tp, args.mrr)
    elif args.wov:
        evaluate(args.o, data_loading_wov['name'], tasks[args.t], args.tp, args.mrr)
    else:
        evaluate(args.o, data_loading_comb['name'], tasks[args.t], args.tp, args.mrr)


def infer(args):
    from type4py.deploy.infer import infer_main
    setup_logs_file(args.m, 'infer')
    infer_main(args.m, args.f)

# add projects-based infer function for command "infer_project"
'''
project-based CLI command includes three approaches:
-t4py : type4py model
-t4pyre: typ4py + pyre
-t4pyright: type4py + pyright
'''
def infer_project(args):
    approach_list = {"t4py", "t4pyre", "t4pyright"}
    if args.a in approach_list:
        from type4py.deploy.infer_project import infer_project_main
        setup_logs_file(args.m, 'infer_project')
        infer_project_main(args.m, args.p, args.o, args.a, args.split)
    else:
        raise InferApproachNotFound

def main():
    arg_parser = argparse.ArgumentParser()
    sub_parsers = arg_parser.add_subparsers(dest='cmd')

    # Extract phase
    extract_parser = sub_parsers.add_parser('extract')
    extract_parser.add_argument('--c', '--corpus', required=True, type=str, help="Path to the Python corpus or dataset")
    extract_parser.add_argument('--o', '--output', required=True, type=str, help="Path to store processed projects")
    extract_parser.add_argument('--d', '--deduplicate', required=False, type=str, help="Path to duplicate files")
    extract_parser.add_argument('--w', '--workers', required=False, default=4, type=int,
                                help="Number of workers to extract functions from the input corpus")
    extract_parser.add_argument('--l', '--limit', required=False, type=int,
                                help="Limits the number of projects to be processed")
    extract_parser.set_defaults(func=extract)

    # Preprocess phase
    preprocess_parser = sub_parsers.add_parser('preprocess')
    preprocess_parser.add_argument('--o', '--output', required=True, type=str, help="Path to processed projects")
    preprocess_parser.add_argument('--l', '--limit', required=False, type=int,
                                   help="Limits the number of projects to be processed")
    preprocess_parser.add_argument('--rvth', '--random-vth', default=False, action="store_true",
                                   help="Apply available type hints with a probability [Default=0.5] *ONLY FOR PRODUCTION*")
    preprocess_parser.set_defaults(func=preprocess)

    # Vectorize phase
    vectorize_parser = sub_parsers.add_parser('vectorize')
    vectorize_parser.add_argument('--o', '--output', required=True, type=str, help="Path to processed projects")
    vectorize_parser.set_defaults(func=vectorize)

    # Learning phase
    learning_parser = sub_parsers.add_parser('learn')
    learning_parser.add_argument('--o', '--output', required=True, type=str, help="Path to processed projects")
    learning_parser.add_argument('--c', '--complete', default=True, action="store_true", help="Complete Type4Py model")
    learning_parser.add_argument('--woi', default=False, action="store_true", help="Type4py model w/o identifiers")
    learning_parser.add_argument('--woc', default=False, action="store_true", help="Type4py model w/o code contexts")
    learning_parser.add_argument('--wov', default=False, action="store_true",
                                 help="Type4py model w/o visible type hints")
    learning_parser.add_argument('--p', '--parameters', required=False, type=str,
                                 help="Path to the JSON file of model's hyper-parameters")
    learning_parser.add_argument('--v', '--validation', default=False, action="store_true",
                                 help="Evaluating Type4Py on the validation set when training")
    learning_parser.set_defaults(func=learn)

    # Learning phase split
    learning_parser_sep = sub_parsers.add_parser('learns')
    learning_parser_sep.add_argument('--o', '--output', required=True, type=str, help="Path to processed projects")
    learning_parser_sep.add_argument('--c', '--complete', default=True, action="store_true",
                                     help="Complete Type4Py model")
    learning_parser_sep.add_argument('--dt', '--datatype', required=True, type=str,
                                     help="Datatype for training phase")
    learning_parser_sep.add_argument('--p', '--parameters', required=False, type=str,
                                     help="Path to the JSON file of model's hyper-parameters")
    learning_parser_sep.add_argument('--v', '--validation', default=False, action="store_true",
                                     help="Evaluating Type4Py on the validation set when training")
    learning_parser_sep.set_defaults(func=learn_split)

    # Prediction phase
    predict_parser = sub_parsers.add_parser('predict')
    predict_parser.add_argument('--o', '--output', required=True, type=str, help="Path to processed projects")
    predict_parser.add_argument('--c', '--complete', default=True, action="store_true", help="Complete Type4Py model")
    predict_parser.add_argument('--l', '--limit', required=False, type=int,
                                help="Limiting the size of type vocabulary when building type clusters")
    predict_parser.add_argument('--rtc', '--reduced-tc', default=False, action="store_true",
                                help="Use reduced type clusters")
    predict_parser.add_argument('--woi', default=False, action="store_true", help="Type4py model w/o identifiers")
    predict_parser.add_argument('--woc', default=False, action="store_true", help="Type4py model w/o code contexts")
    predict_parser.add_argument('--wov', default=False, action="store_true",
                                help="Type4py model w/o visible type hints")
    predict_parser.set_defaults(func=predict)

    # gen type cluster incremental: predict phase generate type cluster
    predict_parser_gen_cluster = sub_parsers.add_parser('gen_type_clu')
    predict_parser_gen_cluster.add_argument('--o', '--output', required=True, type=str, help="Path to processed projects")
    predict_parser_gen_cluster.add_argument('--dt', '--datatype', required=True, help="Datatype for generating type clusters")
    predict_parser_gen_cluster.set_defaults(func=gen_type_cluster)

    # gen predictions via type cluster: predict phase generate predictions
    predict_parser_gen_pred = sub_parsers.add_parser('predicts')
    predict_parser_gen_pred.add_argument('--o', '--output', required=True, type=str, help="Path to processed projects")
    predict_parser_gen_pred.add_argument('--c', '--complete', default=True, action="store_true", help="Complete Type4Py model")
    predict_parser_gen_pred.set_defaults(func=predict_split)

    # Evaluation phase
    eval_parser = sub_parsers.add_parser('eval')
    eval_parser.add_argument('--o', '--output', required=True, type=str, help="Path to processed projects")
    eval_parser.add_argument('--t', '--task', default="c", type=str,
                             help="Prediction tasks (combined -> c |parameters -> p| return -> r| variable -> v)")
    eval_parser.add_argument('--tp', '--topn', default=10, type=int, help="Report top-n predictions [default n=10]")
    eval_parser.add_argument('--mrr', default=False, action="store_true",
                             help="Calculates MRR for all considered metrics")
    eval_parser.add_argument('--woi', default=False, action="store_true", help="Type4py model w/o identifiers")
    eval_parser.add_argument('--woc', default=False, action="store_true", help="Type4py model w/o code contexts")
    eval_parser.add_argument('--wov', default=False, action="store_true", help="Type4py model w/o visible type hints")
    # eval_parser.add_argument('--c', '--combined', default=True, action="store_true", help="combined prediction task")
    # eval_parser.add_argument('--a', '--argument', default=False, action="store_true", help="argument prediction task")
    # eval_parser.add_argument('--r', '--return', default=False, action="store_true", help="return prediction task")
    # eval_parser.add_argument('--v', '--variable', default=False, action="store_true", help="variable prediction task")
    eval_parser.set_defaults(func=eval)

    # Inference
    infer_parser = sub_parsers.add_parser('infer')
    infer_parser.add_argument('--m', '--model', required=True, type=str, help="Path to the pre-trained Type4Py model")
    infer_parser.add_argument('--f', '--file', required=True, type=str,
                              help="Path to the input source file for inference")
    infer_parser.set_defaults(func=infer)

    # Inference project_base
    infer_parser_pro = sub_parsers.add_parser('infer_project')
    infer_parser_pro.add_argument('--m', '--model', required=True, type=str,
                                  help="Path to the pre-trained Type4Py model")
    infer_parser_pro.add_argument('--p', '--path', required=True, type=str,
                                  help="Path to python projects folder for inference")
    infer_parser_pro.add_argument('--o', '--output', required=True, type=str,
                                  help="Path to store the ml_infer outputs")
    infer_parser_pro.add_argument('--a', '--approach', required=True, type=str,
                                  help="infer approach includes t4py, t4pyre, t4pyright, t4pysa")
    # split according to dataset_split_repo.csv
    infer_parser_pro.add_argument('--split', '--split_file', required=True, type=str,
                                  help="file to store the split of projects")
    infer_parser_pro.set_defaults(func=infer_project)

    # To ONNX format
    onnx_parser = sub_parsers.add_parser('to_onnx')
    onnx_parser.add_argument("--o", required=True, type=str, help="Path to processed projects")
    onnx_parser.set_defaults(func=type4py_to_onnx)

    # Reduce
    reduce_parser = sub_parsers.add_parser('reduce')
    reduce_parser.add_argument("--o", required=True, type=str, help="Path to processed projects")
    reduce_parser.add_argument("--d", default=256, type=int, help="A new dimension for type clusters [Default: 256]")
    reduce_parser.add_argument("--batch", default=False, action="store_true", help="Reduce type clusters in batches")
    reduce_parser.set_defaults(func=reduce_tc)

    args = arg_parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
