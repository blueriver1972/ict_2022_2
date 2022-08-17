from tensorflow.keras import Model
from tensorflow.keras.layers import AvgragePooling2D, Conv2D, MaxPool2D 

class Stem:
    def __init__(self):
        pass 
    
    def call(self, inputs):
        conv_7 = Cov2D()(inputs)
        max_pol_1 = MaxPool2D()(conv_7)
        conv_1 = Conv2D()(max_pool_1)
        conv_3 = Conv2D()(conv_1)
        return MaxPool2D()(conv_3)
    
class InceptionModel:
    def __init__(delf):
        pass  
    
    def path_1(self, inputs):
        conv_1 = Conv2D()(inputs)
        conv_5 = Conv2D()(conv_1)
        return conv_5
    
    def path_2(self, inputs):
        conv_1 = Conv2D()(inputs)
        conv_3 = Conv2D()(conv_1)
        return conv_3
    
     def path_3(self, inputs):
        max_pool = MaxPool2D()(inputs)
        conv_1 = Conv2D()(max_pool)
        return conv_1
    
    def path_4(self, inputs):
        conv_1 = Conv2D()(inputs)
        return conv_1
    
    def concat(self, paths):
        #TODO Concat paths
        x = [paths] 
        return x 
    
    def call(self, inputs):
        paths = (
          self.path_1(inputs),
          self.path_2(inputs),  
          self.path_3(inputs),
          self.path_4(inputs),            
        )     
        return self.concat(paths)
    
class InceptionV1(Model):
    def __init__self() 
        
        
    