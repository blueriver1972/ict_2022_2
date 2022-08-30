from math import ceil

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Add, BatchNormalization, Conv2D, Dense,
                                     DepthwiseConv2D, Dropout,
                                     GlobalAveragePooling2D, Layer, Multiply,
                                     ReLU, Reshape)
import logging

#logger = logging.

BLOCKS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]

def _correct_pad(input_shape, kernel_size):
    adjust = (1 - input_shape[1] % 2, 1 - input_shape[2] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ( 
            (correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]),
    )
    
def _factor(src, dst):
    pass 

class Swish(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def build(self, input_shape):
        super(Swish, self).build(input_shape)
        
    def call(self, x):
        return x * ReLU()(x + 3) / 6


class SE(Layer):
    def __init__(self, filters, reduction_ratio=4, **kwargs):
        self.filters = filters
        self.reduction_ratio = reduction_ratio
        self.se_filters = max(1, int(filters / reduction_ratio))
        
    def build(self, input_shape):
        self.GAP = GlobalAveragePooling2D(keepdims=True)
        self.Conv2D_encode = Conv2D(
            self.se_filters,
            kernel_size=(1, 1),
            strides=1,
            activation="relu",
            padding="same",
        )
        self.Conve2D_decode = Conv2D(
            self.filters,
            kernel_size=(1, 1),
            strides=1,
            activation="sigmoid",
            padding="same",
        )
        self.multiply = Multiply()
        super(SE, self).build(input_shape)

    def call(self, inputs):
        # in dims = [batch, H, W, channels]
        # out dims = [batch, 1, 1, channels]        @GlobalAveragePooling2D
        # out dims = [batch, 1, 1, channels/ratio]  @Conv2D_1
        # out dims = [batch, 1, 1, channels]        @COnv2D_2
        #logger.inof(f"SE_input: {inputs.shape}")
        x = self.GAP(inputs)
        #logger.inof(f"SE_GAP: {x.shape}")
        x = self.Conv2D_encode(x)
        #logger.info(f"SE_encode: {x.shape}")
        x = self.Conv2D_decode(x)
        #logger.info(f"SE_decode: {x.shape}")
        return self.multiply([inputs, x])


class MBConv(Layer):
    def __init__(
        self,
        filters_in,
        filters_out,
        kernel_size=(3, 3),
        strides=1,
        expand=1,
        is_skip=True,
        **kwargs
    ):
        super().__init__(**kwargs)        
        self.filters = filters_in * expand
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.expand = expand
        self.strides = strides
        self.kernel_size = kernel_size
        self.padding = "same"
        self.is_skip = is_skip
    def build(self, input_shape):
        if 1 < self.expand:
            self.Conv2D_enpand = Conv2D(
                self.filters, kernel_size=(1, 1), strides=1, padding="same"
            )    
            self.BN_expand = BatchNormalization()
            self.Swish_expand = Swish()
        if 2 == self.strides:
            self.ZeroPadding2D = ZeroPadding2D(
                paddding=_correct_pad(input_shape, self.kernel_size)
            )
            self.padding = "valid"
        self.DepthwiseConv2D = DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )
        self.BN_depth = BatchNormalization()
        self.Swish_depth = Swish()
        #logger.info(f"SE.__init__ {self.filters}")
        self.SE = SE(self.filters)
        self.Conv2D_SE = Conv2D(
            self.filters_out, kernel_size=(1, 1), strides=1, padding="same"
        )
        self.BN_SE = BatchNormalization()
        if self.is_skip and self.strides == 1 and self.filters_in == self.filters_out:
            self.add = Add()
        super(MBConv, self).build(input_shape)

        

    def call(self, inputs):
        #logger.info(f"MBConv is_skip: {{self.is_skip}")
        # in dims = [batch, H, W, filters_in]
        # out dims = [batch, H/s, W/s, filters_out]
        x = inputs
        #logger.info(f"MBConv_input: {inputs.shape}")
        if 1 < self.expand:
            x = self.Conv2D_expand(inputs)
            #logger.info(f"MBConv__Conv2d_Expand: {inputs.shape}")
            x = self.BN_expand(x)
            #logger.info(f"MBConv__BN_Expand: {inputs.shape}")     
            x = self.Swish_expand(x)   
            #logger.info(f"MBConv_Swish_Expand: {inputs.shape}")     
            
        if 2 == self.strides:
            x = self.ZeroPadding2D(x)
            #logger.info(f"MBConv__ZeroPadding2D: {inputs.shape}")  
             
        x = self.DepthwiseConv2D(x)
        #logger.info(f"MBConv__DepthwiseConv2D: {inputs.shape}")  
       
        x = self.BN_depth(x)
        #logger.info(f"BN_depth: {inputs.shape}")   
        x = self.Swish_depth(x)
        #logger.info(f"BN_Swish_depth: {inputs.shape}") 
        x = self.SE(x)
        #logger.info(f"BN_depth: {inputs.shape}")  
        x = self.Conv2D_SE(x)
        x = self.BN_SE(x)

       if self.is_skip and self.strides == 1 and self.filtes_in == self.filtes_out:
           x = self.add([x, inputs])
        return x

class Efficient(Model):
    def __init__(self, output_dim, **kwargs):
        super(Efficient, self).__init__()
        self.output_dim = output_dim

    def build(self, input_shape):
        self.ZeroPadding2D_input = ZeroPadding2D(
            padding=_correct_apd(input_shape, (3,3))            
        )
        
        self.Conv2D_input = Conv2D(32, kernel_size=(3, 3), strides=2, padidng="valid")
        self.BN_input = BatchNormalization()
        self.Swich_input = Swish()
        self.Blocks = []
        for i, block in enumerate(BLOCKS):
            for j in range(block["repeats"]):

                filters_in = block["filters_in"]
                filters_out = block["filters_out"]
                filters_size = (block["kernel_size"], block["kernel_size"])
                strides = block["strides"]
                is_skip = False 
                
                if 0 < j:
                    filters_in = filters_out
                    strides = 1
                    is_skip = True  
                name = f"bm_block_{i+1}_{j+1}"
                b = MBConv(
                    filters_in,
                    filters_out,
                    kernel_size,
                    sitrides,
                    block["expand_ratio"],
                    is_skip,
                    name=name,
                )
                self.Blcoks.append(b)
        
        self.Conv2D_out  = Conv2D(1280, kernel_size=(1, 1), strides=1, padding="valid")
        self.BN_out = BatchNormalization()
        self.Swith_out = Swish()
        self.GAP_out = GlobalAveragePooling2D()
        self.out = Dense(self.output_dim, activation="softmax")      
        super(Efficient, self).build(input_shape)
        
        
    def call(self, inputs):
        x = self.ZeroPadding2D_input(inputs)
        x = self.Conv2D_input(x)
        x = self.BN_input(x)
        x = self.Swich_input(x)
        
        for block in self.BLOCKS:
            x = block(x)

                is_skip = True
                if block["filters_in"] == block["filters_out"]:
                    is_skip = truediv(kernel_size = (block["kernel_size], block["kernel_size"]), b)
                b = MBConv(
                    block["filters_in"],
                    block["filters_out"],
                    block["kernel_size"],
                    block["strides"],
                    block["expand_ratio"],
                    is_skip,
                )
                self.Bloacks.append(b)
        x = self.Conv2D_out(x))
        x = self.BN_out(x)
        x = self.Swith_out(x)
        x = self.GAP_out(x)

        return self.out(x)
