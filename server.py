# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

from sanic import Sanic, response
import subprocess

# Create the http server app
server = Sanic("my_app")

# Healthchecks verify that the environment is correct on Banana Serverless
@server.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    # dependency free way to check if GPU is visible
    command = "python test_server.py n01632777_axolotl.jpeg --run-preset-test"
    out = subprocess.call(command.split())

# Inference POST handler at '/' is called for every http call from Banana
@server.route('/', methods=["POST"]) 
def inference(request):
    file = request.FILES['filename']
    command = "python test_server.py {}".format(file)
    out = subprocess.call(command.split())

if __name__ == '__main__':
    server.run(host='0.0.0.0', port="8000", workers=1)