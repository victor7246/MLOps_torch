import subprocess
from test_onnx import load_onnx_model, get_prediction

from convert_to_onnx import convert_torch_to_onnx

def get_inference(image_path):
    command = "python convert_torch_to_onnx.py pytorch_model_weights.pth"
    subprocess.call((command.split())
    
    ort_session = load_onnx_model("model.onnx")
    return get_prediction(image_path, ort_session)


