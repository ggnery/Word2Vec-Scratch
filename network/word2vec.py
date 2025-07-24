import math
import numpy as np
from .activations import softmax
from .losses import categorical_crossentropy

class Word2Vec:
    def __init__(self, embedding_size: int, vocabulary_size: int, eta: float):
        self.w1 = np.random.randn(embedding_size, vocabulary_size)
        self.w2 = np.random.randn(vocabulary_size, embedding_size) 
        self.eta = eta

    def forward(self, x: np.ndarray) -> dict[str, np.ndarray]:
        h = self.w1 @ x  
        a = self.w2 @ h
        y = softmax(a)
        
        return {"output": y, "input": x, "hidden": h, "activation": a}
        
    def backward(self, output_gradient: np.ndarray, cache: dict[str, np.ndarray]):
        delta_w2 = output_gradient @ cache["hidden"].T 
        delta_h = self.w2.T @ output_gradient
        delta_w1 = delta_h @ cache["input"].T
        
        #SGD
        self.w1 -= self.eta * delta_w1
        self.w2 -= self.eta * delta_w2
    
    def train(self, input, output, epochs):
        for epoch in range(epochs):
            loss = 0
            for i, (x, y) in enumerate(zip(input, output)):
                cache = self.forward(x)
                output_gradient = cache["output"] - y
                loss += categorical_crossentropy(y, cache["output"])
                self.backward(output_gradient, cache)
            print(f"Epoch {epoch}/{epochs}: Loss -> {loss/i}")