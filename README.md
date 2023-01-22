# MLOps

Sample MLOps project for deep learning model deployment

### How to run

Convert Pytorch to ONNX

```
python convert_to_onnx.py pytorch_model_weights.pth

```

Run test cases

```
python test_onnx.py n01632777_axolotl.jpeg
```

Run inference with Banana

```
python test_server.py --image-path n01440764_tench.jpeg
```
