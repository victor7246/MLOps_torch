import argparse
import os
from test_onnx import load_onnx_model, get_prediction

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

    ort_session = load_onnx_model("model.onnx")
    img_out_y = get_prediction(image_path, ort_session)

    print ("Predicted class is {}".format(img_out_y))

    if args.run_preset_test == True:
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