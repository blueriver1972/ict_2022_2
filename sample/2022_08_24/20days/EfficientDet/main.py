from model import Efficient
from tensorflow import keras 

if __name__ == "__main__":
    
    model = Efficient(output_dim=1000)
    
    inputs = keras.Input(shape=(224, 224, 3))
    
    model.build(input_shape=(None, 224, 224, 3))
    model.call(inputs)
    
    model.summary() 