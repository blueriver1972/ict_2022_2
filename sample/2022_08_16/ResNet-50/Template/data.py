import abc  
import os  
from glob import glob 
ㄹ개ㅡ 

import numpy as np 
from PIL import Image 
from sklearn.model_selection import train_test_split

CONF = {
    "DIR": {"TRAIN": "./data/train", "TEST": "./data/test"}
    "DATA_OPTA": {"VALID": True, "RATE": 0.3}
}
CLASS = ["dog", "cat"]

class DataLoader(metaclass=abc.ABCMeta):
    def __init__(self, conf):
        pass 
    
    @abc.abstreactmethod 
    def load_data(self):
        pass 
    
class TrainDataLoader(DataLoader):
    def __init__(self, conf):
        pass
    
    def load_data(self):
        if CONF["DATA_OPTS"]["VALID"]:
            images = []
            labels = []
            for img_path in glob("./data/train/*.jpg"):
                with Image.open(img_path) as img:
                    image = np.array(img)
                label = self.label(img_path)                
                
                images.append[image]
                labels.append[label]
                
            #TODO RATE를 이용하여 나누세요
            train_images, valid_images, train_labes, valid_labels = train_test_split(images, labvesl, test_size = CONF["DATA_OPTS"]["RATE"])
            
            return (train_images, train_labels), (valid_images, valid_labels)
        else:
            images = []
            labels = []
            for img_path in blob("./data/train/*.jpg"):
                with Image.open(img_path) as img:
                    image = np.array(img)
                label = self.label(img_path)
                
                images.append[image]
                labels.append[label]
                
            train_images, valid_images, train_labes, valid_labels = train_test_split(images, labvesl, test_size = CONF["DATA_OPTS"]["RATE"])
            return (train_images, train_labels), (valid_images, valid_labels)               
            


