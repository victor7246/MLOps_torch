import sys
import os
import onnx
import onnxruntime
import numpy as np

from utils import read_image, preprocess_image, to_numpy

def load_onnx_model(path):
    ort_session = onnxruntime.InferenceSession(path)

    return ort_session

def get_prediction(im, ort_session):
    ort_inputs = {ort_session.get_inputs()[0].name: im}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    return img_out_y.argmax(-1)[0]

if __name__ == '__main__':
    image_path = sys.argv[1]
    ort_session = load_onnx_model("model.onnx")
    im = preprocess_image(read_image(image_path))

    img_out_y = get_prediction(im, ort_session)

    if os.path.basename(image_path) == 'n01440764_tench.jpeg':
        try:
            assert img_out_y == 0
            print ("Test case for n01440764_tench passed!")
        except AssertionError:
            print ("Test case for n01440764_tench failed!")
        
    elif os.path.basename(image_path) == 'n01632777_axolotl.jpeg':
        try:
            assert img_out_y == 26
            print ("Test case for n01632777_axolotl passed!")
        except AssertionError:
            print ("Test case for n01632777_axolotl failed!")