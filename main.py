from dataset import Dataset
from network import Word2Vec

def main():
    dataset = Dataset("dataset/text8", 50)
    x, y = dataset.generate_training_data(7)
        
    model = Word2Vec(300, dataset.vocabulary_size, 0.1)
    model.train(x, y, 100)
    
if __name__ == "__main__":
    main()