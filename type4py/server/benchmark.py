# A script to benchmark the server with a given file

from argparse import ArgumentParser
from libsa4py.utils import read_file
import requests
import concurrent
import time

T4PY_API_URL = "https://type4py.com/api/predict?tc=0"

def run_benchmark(args):
    def req_post(f):
        return requests.post(T4PY_API_URL, f)

    f_read = read_file(args.f)
    start_t = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        res = [executor.submit(req_post, f_read) for i in range(args.r)]
        concurrent.futures.wait(res)
    
    print(f"Processed {args.r} reqeusts in {time.time()-start_t:.2f} sec.")



if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Benchmarking server")
    arg_parser.add_argument("--f", required=True, type=str, help="Path to a source code file")
    arg_parser.add_argument("--r", required=False, default=25, type=int, help="Number of requests to send")
    arg_parser.set_defaults(func=run_benchmark)

    args = arg_parser.parse_args()
    args.func(args)
