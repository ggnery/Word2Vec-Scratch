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
        loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_samples = len(input)
            
            for x, y in zip(input, output):
                cache = self.forward(x)
                output_gradient = cache["output"] - y
                epoch_loss += categorical_crossentropy(y, cache["output"])
                self.backward(output_gradient, cache)
            
            avg_loss = epoch_loss / num_samples
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}: Loss -> {avg_loss:.6f}")
        
        return loss_history
    
    def get_embedding(self, one_hot_word):
        return self.forward(one_hot_word)["hidden"]
    
    def cosine_similarity(self, embedding1, embedding2):
        # Flatten the embeddings to 1D vectors
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.sqrt(np.sum(np.square(embedding1)))
        norm2 = np.sqrt(np.sum(np.square(embedding2)))
        return dot_product/ (norm1*norm2)