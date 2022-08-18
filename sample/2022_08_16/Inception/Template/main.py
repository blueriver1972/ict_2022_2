from data import DataLoader
from model import INceptionV1
from tensorflow.keras.datasets import cifar10
if __name__ == "__main__":
    
    
    dataset = DataLoader()
    print(dataset.datasets)
    
    model = IncetptionsV1(output_dim = dataset.n_classes)
    
    
    