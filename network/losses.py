import numpy as np

def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -np.sum(y_true * np.log(y_pred))