import random
from turtle import distance
import matplotlib.pyplot as plt
from dataset import Dataset
from network import Word2Vec
import numpy as np
def main():
    dataset = Dataset("dataset/text8", 1000)
    x, y = dataset.generate_training_data(10)
        
    model = Word2Vec(300, dataset.vocabulary_size, 0.1)
    loss_history = model.train(x, y, 20)
    
    distances = {}
    ramdom_word = np.random.choice(list(dataset.vocabulary.keys()))
    #ramdom_word = "king"
    ramdom_word_embedding = model.get_embedding(dataset.one_hot_encode(ramdom_word))
    for word in dataset.vocabulary.keys():
        if word != ramdom_word:
            word_embedding = model.get_embedding(dataset.one_hot_encode(word))
            distances[word] = model.cosine_similarity(ramdom_word_embedding, word_embedding) 
    
    k = 5
    distances = sorted(distances.items(), key=lambda x: x[1], reverse=True)
    top_k = distances[:k]
    
    print(f"\nTop {k} most similar words to '{ramdom_word}':")
    print("-" * 40)
    for word, similarity in top_k:
        print(f"{word}: {similarity:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2)
    plt.title('Word2Vec Training Loss Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()
    
if __name__ == "__main__":
    main()