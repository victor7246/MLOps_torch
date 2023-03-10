import argparse
import os
import json
import numpy as np
from utils import read_image, preprocess_image

import banana_dev as banana

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Test server for Banana deployment")
    parser.add_argument(
        "--image-path",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--run-preset-test", action="store_true", default=False, help="Enables preset testing"
    )

    args = parser.parse_args()

    image_path = args.image_path

    image = preprocess_image(read_image(image_path))

    data = json.dumps({'image': image}, cls=NumpyEncoder)

    api_key = "6386c6ad-ef57-4461-a44e-bab245e0fe40"
    model_key = "b176f86c-20bc-4914-a829-b9955e2aec3b" #"5bdf58a0-3b55-4d0b-b40b-37da861aff1c"

    print ("Running prediction in banana")
    out = banana.run(api_key, model_key, data)

    print (out)

    