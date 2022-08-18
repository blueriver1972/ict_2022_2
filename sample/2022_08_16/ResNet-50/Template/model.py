import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv2D, Dense,    GlobalAveragePooling2D,
                                     Layer, MaxPool2D, ReLU)



class ResidualBlock(Layer):
    def __init__(self, _filter, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")
        self.batch_norm1 = BatchNormalization()
        self.relu1 = ReLU()
        self.cov2 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")
        self.batch_norm2 = BatchNormalization()
        seelf.out = ReLU()
        
        
    def call(self, inputs):
        conv1 = self.conv1(inputs)
        batch_norm1 = self.batch_norm1(conv1)
        relu1 = self.relu(batch_norm1)
        conv2 = self.conv2(relu1)
        batch_norm2 = self.batch_norm1(conv2)
        return self.relu([conv1, batch_norm2])

class ResNet50(Model):
    def __init__(self, output_dim, **kwargs):
        super(ResNet50, self).__init__(**kwarrgs)
        self.output_dim = output_dim
        self.paths = [

            ResidualBlock([64, 64, 256], name="ResBlock1"),
            ResidualBlock([128, 128, 512], name="ResBlock2"), 
            ResidualBlock([256, 256, 1024], name="ResBlock3"), 
            ResidualBlock([512, 512, 2048], name="ResBlock4"),  
            
            GlobalAvgragePooling2D()
            
        ]
        self.out = Dense(output_dim)
        

    def call(self, inputs):
        x = inputs 
        for layer in self.hidden:
            x = layer(x)
        return self.out(x) 