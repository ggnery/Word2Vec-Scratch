import re
import numpy as np

def softmax(x: np.ndarray):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=0, keepdims=True)
