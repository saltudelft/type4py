"""
Converts the pre-trained Type4Py model to ONNX
"""

from os.path import join
from type4py import logger
from type4py.data_loaders import to_numpy
import torch
import numpy as np
import onnx
import onnxruntime

logger.name = __name__
BATCH_SIZE = 1
BATCH_SIZE_TEST = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def type4py_to_onnx(args):
    type4py_model = torch.load(join(args.o, "type4py_complete_model.pt")).model
    type4py_model.eval()
    logger.info("Loaded the pre-trained Type4Py model")

    x_id, x_tok, x_avl = torch.randn(BATCH_SIZE, 31, 100, requires_grad=True).to(DEVICE), torch.randn(BATCH_SIZE, 21, 100, requires_grad=True).to(DEVICE), \
                         torch.randint(low=0, high=2, size=(BATCH_SIZE, 1024), dtype=torch.float32, requires_grad=True).to(DEVICE)

    # For testing dynamic axes when using the exported model
    x_id_test, x_tok_test, x_avl_test = torch.randn(BATCH_SIZE_TEST, 31, 100, requires_grad=True).to(DEVICE), \
                                        torch.randn(BATCH_SIZE_TEST, 21, 100, requires_grad=True).to(DEVICE), \
                                        torch.randint(low=0, high=2, size=(BATCH_SIZE_TEST, 1024), dtype=torch.float32, requires_grad=True).to(DEVICE)
    
    t_out_test = type4py_model(x_id_test, x_tok_test, x_avl_test)
    
    torch.onnx.export(type4py_model, (x_id, x_tok, x_avl), join(args.o, "type4py_complete_model.onnx"),
                      export_params=True, do_constant_folding=True, input_names = ['id', 'tok', 'avl'], output_names = ['output'],
                      dynamic_axes={'id': {0 : 'batch_size'}, 'tok': {0 : 'batch_size'}, 'avl': {0 : 'batch_size'},
                      'output': {0 : 'batch_size'}})
    
    logger.info("Exported the pre-trained Type4Py model to an ONNX model")

    type4py_onnx_m = onnx.load(join(args.o, "type4py_complete_model.onnx"))
    onnx.checker.check_model(type4py_onnx_m)

    ort_session = onnxruntime.InferenceSession(join(args.o, "type4py_complete_model.onnx"), providers=['CUDAExecutionProvider'])
    ort_inputs =  {ort_session.get_inputs()[0].name: to_numpy(x_id_test),
                   ort_session.get_inputs()[1].name: to_numpy(x_tok_test),
                   ort_session.get_inputs()[2].name: to_numpy(x_avl_test)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(t_out_test), ort_outs[0], rtol=1e-03, atol=1e-05)
    logger.info("The exported Type4Py model has been tested with ONNXRuntime, and the result looks good!")