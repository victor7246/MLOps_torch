# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git ffmpeg

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ADD model.onnx .
ADD *.py .

# We add the banana boilerplate here
# ADD server.py .

# EXPOSE 8000

# CMD python3 -u server.py
