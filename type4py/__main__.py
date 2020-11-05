from type4py.extract_pipeline import Pipeline
from type4py.preprocess import preprocess_ext_fns
from type4py.vectorize import vectorize_args_ret
import argparse

def extract(args):
    p = Pipeline(args.c, args.o)
    p.run(args.w, args.l)

def preprocess(args):
    preprocess_ext_fns(args.o)

def vectorize(args):
    vectorize_args_ret(args.o)

def learn(args):
    if args.a or args.r:
        args.c = False
    
    print(args)


def main():
    arg_parser = argparse.ArgumentParser()
    sub_parsers = arg_parser.add_subparsers(dest='cmd')

    # Extract phase
    extract_parser = sub_parsers.add_parser('extract')
    extract_parser.add_argument('--c', '--corpus', required=True, type=str, help="Path to the Python corpus or dataset")
    extract_parser.add_argument('--o', '--output', required=True, type=str, help="Path to store processed projects")
    extract_parser.add_argument('--w', '--workers', required=False, default=4, type=int, help="Number of workers to extract functions from the input corpus")
    extract_parser.add_argument('--l', '--limit', required=False, type=int, help="Limits the number of projects to be processed")
    extract_parser.set_defaults(func=extract)

    # Preprocess phase
    proprocess_parser = sub_parsers.add_parser('preprocess')
    proprocess_parser.add_argument('--o', '--output', required=True, type=str, help="Path to processed projects")
    proprocess_parser.set_defaults(func=preprocess)

    # Vectorize phase
    vectorize_parser = sub_parsers.add_parser('vectorize')
    vectorize_parser.add_argument('--o', '--output', required=True, type=str, help="Path to processed projects")
    vectorize_parser.set_defaults(func=vectorize)

    # Learning phase
    learning_parser = sub_parsers.add_parser('learn')
    learning_parser.add_argument('--o', '--output', required=True, type=str, help="Path to processed projects")
    learning_parser.add_argument('--c', '--combined', default=True, action="store_true", help="combined prediction task")
    learning_parser.add_argument('--a', '--argument', default=False, action="store_true", help="argument prediction task")
    learning_parser.add_argument('--r', '--return', default=False, action="store_true", help="return prediction task")
    learning_parser.add_argument('--p', '--parameters', required=False, type=str, help="Path to the JSON file of model's hyper-parameters")
    learning_parser.set_defaults(func=learn)

    args = arg_parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()