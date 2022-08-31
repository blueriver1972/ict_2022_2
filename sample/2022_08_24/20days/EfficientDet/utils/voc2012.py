import os
import typing
import xml.etree.ElementTree as ET 
import cv2
import random 

from utils import BBox 
from glob import glob



ROOT_DIR = "../data/VOC2012/"
ANNOTATIONS_DIR = ROOT_DIR + "Annotations/"
IMG_DIR = ROOT_DIR + "JPEGImages"

Bbox = tuple[tuple[int, int], tuple[int, int]]
Img = list[list[list[int]]]
#pixel  = [r, g, b]
#row = list[pixel]
#col = list[row]

class VOC:
    def __init__(self, root_dir=None):
        self.root_dir = root_dir
        self.annotations_dir = root_dir + "Annotations/"
        self.img_dir = root_dir + "JPEGImages/"
        
    def img_n_bbox(self, fname_img) -> tuple[Img, list[Bbox]]:
        xml = self.get_xml(fname_img)
        
        img_path = voc.img_dir + xml.findtext("filename")
        img = cv2.imread(img_path)
        
        bboxs = []
        for obj in xml.findall("object"):
            bbox = obj.find("bndbox")
            start = int(bbox.findtext("xmin")), int(bbox.findtext("ymin"))
            end = int(bbox.findtext("xmax")), int(bbox.findtext("ymax"))
            bboxs.append((start, end))   
        
        return img, bboxs
    
    
    def get_xml(self, xml_name):
        self.tree = ET.parse(self.annotations_dir+xml_name)
        return self.tree.getroot()

    
if __name__ == "__main__":
    voc = VOC(ROOT_DIR)
    bbox = BBox()

    globs = glob(ANNOTATIONS_DIR + "*")
    #print(globs[0])
    
    for rn in range(10):
        #file_name = os.path.basename(random.choice(globs))
        img, bboxs = voc.img_n_bbox(os.path.basename(random.choice(globs)))   
        #print(bboxs)
        #print(file_name)
        bbox.show(img, bboxs)
    
    #img, bboxs = voc.img_n_bbox("2007_000032.xml")   
    
    
    