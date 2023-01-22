from typing import Tuple, List

import cv2
import numpy as np
import torch
from torchvision import transforms

def read_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def preprocess_image(image : np.array, size: Tuple[int] =(224,224), \
        normalize_mean: List[float] = [0.485, 0.456, 0.406], normalize_std: List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor:
    
    image = cv2.resize(image, size,
               interpolation = cv2.INTER_LINEAR)
    image = image/255.0
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(normalize_mean,normalize_std)])

    return transform(image)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
