from model import Efficient
from tensorflow import keras 

def resample_test(model):
    layer_3 = model.get_layer("mb_block_3_3")
    shape_3 = layer_3.output.shape
    print(shape_3)
    
    layer_4 = model.get_layer("mb_block_4_3")
    shape_4 = layer_4.output.shape
    print(shape_4)
    
    layer_5 = model.get_layer("mb_block_5_3")
    shape_5 = layer_5.output.shape
    print(shape_5)
    
    layer_6 = model.get_layer("mb_block_6_4")
    shape_6 = layer_6.output.shape
    print(shape_6)
    
    layer_7 = model.get_layer("mb_block_7_1")
    shape_7 = layer_7.output.shape
    print(shape_7)


    # NOTE Connect -> [7], [7, 6], [6, 5], [5, 4], [4, 3]
    
    # NOTE [(7, 7, 320)] -> (7, 7, 64)
    # NOTE [(7, 7, 64), (7, 7, 192)] -> (7, 7, 64) 
    # NOTE [(7, 7, 64), (14, 14, 112)] -> (14, 14, 64) 
    # NOTE [(14, 14, 64), (14, 14, 80)] -> (14, 14, 64)     
    # NOTE [(14, 14, 64), (28, 28, 80)] -> (14, 14, 64) 
    
    # relayer7 = ResampleFeatureMap(shape_7) 
    
    # model = keras.Sequential(
    #     # [keras.layers.InputLayer(nput_shape = (7, 7, 320)), ResampleFeatrueMap([shape_7], filters=64)]
    #     [
    #         keras.layers.InputLayer(input_shape=(7, 7, 64, 7, 7, 192)),
    #         ResampleFeatureMap([shape_7, shape_6], filters=64),
    #         ]
    #     ]
    # )   
    
    model = keras.Suquential([keras.layers.InputLayer(input_shape=shape_7[1:])])
    relayer7 = ResampleFeatureMap([shape_7], filters=64) 
    #relayer6 = ResampleFeatureMap([shape_7], filters=64) 
    model
    model.summary()  
    
if __name__ == "__main__":
    
    # model = Efficient(output_dim=1000)
    
    # inputs = keras.Input(shape=(224, 224, 3))
    
    # model.build(input_shape=(None, 224, 224, 3))
    # model.call(inputs)
    
    # model.summary() 
    
    inputs = keras.Input(shape=(224, 224, 3))
    backborn = Efficient()(inputs)
    class_out = keras.layers.Dense(1000)(backborn)
    bbox_out = eras.layers.Dense(4)(backborn)
    
    model = keras.Model(input=inputs, output=[class_out, bbox_out])
    
    model.compile(
        loss=["sparse_catrgorical_crossentopy", "mse"], 
        loss_weights=[0.8, 0.2], 
        optimizer="Adam",                 
     )
    