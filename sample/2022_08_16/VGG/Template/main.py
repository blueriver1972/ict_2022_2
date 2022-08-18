from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from model import VGG

if __name__ == "__main__":
    #TODO Data Load
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    
    train_images = train_images/256
    model = VGG()
    
   
    model.compile(
        optimizer = Adam(lr=0.001),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    

   # model.summary()

    # TODO: Training
    hist = model.fit(train_images, train_labels, epochs=10)
