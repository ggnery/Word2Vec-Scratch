import numpy as np

class Dataset:
    def __init__(self, data_path, dataset_slice = 500):
        f = open(data_path, "r")
        self.data = f.read().split(" ")[:dataset_slice]
        self.vocabulary = self.create_vocabulary()
        self.vocabulary_size = len(self.vocabulary)
    
    def create_vocabulary(self):
        unique_words = list(set(self.data))
        return {word: i for i, word in enumerate(unique_words)}
    
    def one_hot_encode(self, word):
        one_hot = np.zeros((self.vocabulary_size, 1))
        one_hot[self.vocabulary[word]] = 1
        return one_hot
        
    def generate_training_data(self, window_size):
        x = []
        y = []
        for window_pos in range(window_size//2, len(self.data) - window_size//2):
            window_words = self.data[window_pos - (window_size//2) : window_pos + (window_size//2)]
            key_word = window_words[len(window_words)//2]
            
            for target_word in window_words: 
                if target_word != key_word:
                    x.append(self.one_hot_encode(key_word))
                    y.append(self.one_hot_encode(target_word))
            
        return x, y