"""
Converts the pre-trained Type4Py model to ONNX
"""

from argparse import ArgumentParser
from pathlib import Path
from os.path import join
from type4py import logger
from type4py.data_loaders import to_numpy
import torch
import numpy as np
import onnx
import onnxruntime

logger.name = __name__
BATCH_SIZE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def type4py_to_onnx(args):
    #id, tok, a = load_combined_valid_data("/home/amir/MT4Py_typed_full/")
    #print("ID", id.shape, "TOK", tok.shape, "avl", a.shape)
    type4py_model = torch.load(args.m).model
    type4py_model.eval()
    logger.info("Loaded the pre-trained Type4Py model")

    x_id, x_tok, x_avl = torch.randn(BATCH_SIZE, 31, 100, requires_grad=True).to(DEVICE), torch.randn(BATCH_SIZE, 21, 100, requires_grad=True).to(DEVICE), \
                         torch.randint(low=0, high=2, size=(BATCH_SIZE, 1024), dtype=torch.float32, requires_grad=True).to(DEVICE)
    
    t_out = type4py_model(x_id, x_tok, x_avl)
    
    torch.onnx.export(type4py_model, (x_id, x_tok, x_avl), join(Path(args.m).parent, Path(args.m).stem+".onnx"),
                      export_params=True, do_constant_folding=True, input_names = ['id', 'tok', 'avl'], output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    logger.info("Exported the pre-trained Type4Py model to an ONNX model")

    type4py_onnx_m = onnx.load(join(Path(args.m).parent, Path(args.m).stem+".onnx"))
    onnx.checker.check_model(type4py_onnx_m)

    ort_session = onnxruntime.InferenceSession(join(Path(args.m).parent, Path(args.m).stem+".onnx"))
    ort_inputs =  {ort_session.get_inputs()[0].name: to_numpy(x_id),
                   ort_session.get_inputs()[1].name: to_numpy(x_tok),
                   ort_session.get_inputs()[2].name: to_numpy(x_avl)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(t_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    logger.info("The exported Type4Py model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Converting Type4Py to ONNX")
    arg_parser.add_argument("--m", required=True, type=str, help="Path to the pre-trained Type4Py model")

    arg_parser.set_defaults(func=type4py_to_onnx)

    args = arg_parser.parse_args()
    args.func(args)