import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv2D, Dense,    GlobalAveragePooling2D,
                                     Layer, MaxPool2D)

class Stem(Layer):
    def __init__(self, **kwargs):
        super(Stem, sefl).__init__(**kwargs)
        self.hidden = [
            Conv2D(filters=64, 
                   kernel_size=(7, 7),
                   strides=(2, 2),
                   activation="relu",
                   padding="same",
            ),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"),
        ] 
    
    def call(self, inputs):
        x = inputs 
        for layer in self.hidden:
            x = layer(x)
        return X

class ResNet50Model(Layer):
    def __init__(self, filter, strider):
        super(ResNet50Model, self).__init__()

        self.paths = [         
           [Conv2D(filters=filter, kernel_size=(3, 3)), strides=(strider, strider), padding="same",],
        ] 
        
    def call(self, input):
        x = path_outs 
        for path in self.paths:
            for layer in path:
                x = layer(x)
            path_outs.append(x)
        return self.depth_concat(path_outs, axis =3) 

class ResNet50(Model):
    def __init__(self, output_dim, **kwargs):
        super(ResNet50, self).__init__(**kwarrgs)
        self.output_dim = output_dim
        self.paths = [
            Stem(name="stem"),
            #INFO [3x3, 3x3, 3x3, 3x3, , 3x3_reduce, 3x3, 5x5_reduce, 5x5, 1x1_pool]
            ResNet50Model(64, 1),
            ResNet50Model(64, 1),  
            ResNet50Model(64, 1),
            ResNet50Model(64, 1),
             
            ResNet50Model(128, 2),
            ResNet50Model(128, 1),
            ResNet50Model(128, 1),
            ResNet50Model(128, 1),
            
            GlobalAvgragePooling2D()
            
        ]
        self.out = Dense(10)
        

    def call(self, inputs):
        x = inputs 
        for layer in self.hidden:
            x = layer(x)
        return self.out(x) 