from dataset import Dataset

def main():
    dataset = Dataset("dataset/text8")
    x, y = dataset.generate_training_data(3)
    print(x)
    
if __name__ == "__main__":
    main()