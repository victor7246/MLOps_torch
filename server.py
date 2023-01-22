# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

from sanic import Sanic, response
import subprocess
import numpy as np
import torch

from utils import to_numpy
from test_onnx import get_prediction, load_onnx_model

# Create the http server app
server = Sanic("my_app")

# Healthchecks verify that the environment is correct on Banana Serverless
@server.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    # dependency free way to check if GPU is visible
    command = "python test_onnx.py n01632777_axolotl.jpeg --run-preset-test"
    out = subprocess.call(command.split())

# Inference POST handler at '/' is called for every http call from Banana
@server.route('/', methods=["POST"]) 
def inference(request):
    data = request.json
    arr = np.array(data['image'])
    ort_session = load_onnx_model("model.onnx")
    print ("Running inference")
    out = get_prediction(arr.astype(float),ort_session)
    
    return response.json({'predicted label': out})

if __name__ == '__main__':
    server.run(host='0.0.0.0', port="8000", workers=1)