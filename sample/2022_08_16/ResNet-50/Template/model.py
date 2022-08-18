import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv2D, Dense,    GlobalAveragePooling2D,
                                     Layer, MaxPool2D, ReLU)



class ResidualUnit(Layer):
    def __init__(self, _filters, _strides=1, **kwargs):
        super(ResidualUnit, self).__init__(**kwargs)
        conv1_filter, conv2_filter = _filters
        
        if self.strides > 1:
            self.conv1 = Conv2D(filters=conv1_filter, kernel_size=(3, 3), strides=(2, 2), padding="same")
        else:
           self.conv1 = Conv2D(filters=conv1_filter, kernel_size=(3, 3), strides=(1, 1), padding="same") 
           
        self.batch_norm1 = BatchNormalization()
        self.relu1 = ReLU()
        self.cov2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")
        self.batch_norm2 = BatchNormalization()
        seelf.out = ReLU()
        
        
    def call(self, inputs):
        if self.strides > 1:
            conv1 = self.conv1(inputs)
            batch_norm1 = self.batch_norm1(conv1)
            relu1 = self.relu(batch_norm1)
            conv2 = self.conv2(relu1)
            batch_norm2 = self.batch_norm1(conv2)
            return self.relu([batch_norm2])
        else:
            conv1 = self.conv1(inputs)
            batch_norm1 = self.batch_norm1(conv1)
            relu1 = self.relu(batch_norm1)
            conv2 = self.conv2(relu1)
            batch_norm2 = self.batch_norm1(conv2)
            return self.relu([conv1 + batch_norm2])

class ResNet50(Model):
    def __init__(self, output_dim, **kwargs):
        super(ResNet50, self).__init__(**kwarrgs)
        self.output_dim = output_dim
        self.conv1 = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same")
        self.max_pool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")
        self.block1 = [
            ResidualUnit([64, 64], name = f"ResBlock1_{i}") for i in range(3)
        ]
        self.block2 = [ResidualUnit([128, 128], name = f"ResBlock2_{i}") for i in range(4)]
        self.block3 = [ResidualUnit([256, 256], name = f"ResBlock3_{i}") for i in range(6)]
        self.block4 = [ResidualUnit([512, 512], name = f"ResBlock4_{i}") for i in range(3)]  
        self.average_pool = GlobalAvgragePooling2D()
        self.out = Dense(output_dim)
        

    def call(self, inputs):
        x = self.conv1(inputs) 
        x = self.max_pool(x)
        for unit in block1:
            x = unit(x)
        for unit in block2:
            x = unit(x)
        for unit in block3:
            x = unit(x)   
        for unit in block4:
            x = unit(x)                                 
        return self.out(x) 