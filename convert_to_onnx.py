import sys
import torch
import onnxruntime
from pytorch_model import Classifier

if __name__ == '__main__':
    
    saved_model_path = sys.argv[1]

    onnx_filename = "model.onnx"

    model = Classifier()
    model.eval()
    model.load_state_dict(torch.load(saved_model_path))
    x = torch.randn(1, 3, 224, 224, requires_grad=True)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    onnx_filename,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})