from type4py.extract_pipeline import Pipeline
import argparse


def main():
    arg_parser = argparse.ArgumentParser()
    
    # Extract phase
    extract_parser = arg_parser.add_subparsers().add_parser('extract')
    extract_parser.add_argument('--c', '--corpus', required=True, type=str, help="Path to the Python corpus or dataset")
    extract_parser.add_argument('--o', '--output', required=True, type=str, help="Path to store processed projects")
    extract_parser.add_argument('--l', '--limit', required=False, type=int, help="Limits the number of projects to be processed")

    args = arg_parser.parse_args()

    p = Pipeline(args.c, args.o)
    p.run(24, args.l)



if __name__ == "__main__":
    main()