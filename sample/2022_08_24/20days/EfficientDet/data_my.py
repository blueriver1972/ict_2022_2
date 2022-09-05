
import re 
from glob import glob 
import cv2
import os

WNIDS = "./data/tiny-imagenet-200/wnids.txt"
WORDS = "./data/tiny-imagenet-200/words.txt"
TRAIN_DIR = "./data/tiny-imagenet-200/train/"

# NOTE [image, (label, bbox)]

def search_class(fd=None, match=""):
    #re = re.compile(match)
    fd.seek(0)
    for line in fd:
        if re.search(match, line):
            # s = line.strip()
            # #print(s)
            # s = s.split("\t")
            # s=s[1]     
            class_name = line.strip().split("\t")[1]
            return class_name
        
        
# INFO {"n02795169": "cat"}
def class_name():
    # {"wnids": "class_name"}
    wnids = open(WNIDS, "r") 
    words = open(WORDS, "r") 
    
    class_names = []
    while True:
        wnid = wnids.readline().strip()
        class_dict = {wnid: search_class(words, wnid)}
        if not wnid:
            break
        class_names.append(class_dict)

    wnids.close()
    words.close()
    return class_names

def findimage(fname):
    namedir = fname.split("_")[0]
    images_texts = glob(TRAIN_DIR+"namedir/*.JPGE")
    img = []
    for img_text in images_texts:
        if img_text == fname:
            img = cv2.imread(os.path.join(namedir, img_text))        
            break 
    return img

def _bbox(bbox_txt) -> list:
    with open(bbox_txt, "r") as fd:
        dir_name = os.path.join(os.path.dirname(bbox_txt), "images")
        bboxes = []
        for line in fd:
            line = line.strip()
            fname, xmin, ymin, xmax, ymax = line.split("\t")
            #print(f"dir : {dir_name}")
            #print(f"filename : {fname}")
            img = cv2.imread(os.path.join(dir_name, fname))
            bbox_dict = {
                "fname": fname, 
                "bbox": ( int(xmin), int(ymin), int(xmax), int(ymax)),
                "image": img,
                #"bbox": ( (xmin), (ymin), (xmax), (ymax))
            }
            bboxes.append(bbox_dict)
    #print(bboxes)
    return bboxes
    
# def _bbox(fd=None) -> list:
#     bboxes = []
#     for line in fd:
#         line = line.strip()
#         fname, xmin, ymin, xmax, ymax = line.split("\t")
        
#         bbox_dict = {
#             "fname": fname, 
#             "bbox": ( int(xmin), int(ymin), int(xmax), int(ymax)),
              #"bbox": ( (xmin), (ymin), (xmax), (ymax))
#         }
#         bboxes.append(bbox_dict)
#     #print(bboxes)
#     return bboxes 


# INFO {"fname": "n02795169_210.JPEG", "bbox":(0, 0, 1, 1)}   
# TODO -> {"fname": "n02795169_210.JPEG", "bbox":(0, 0, 1, 1), "image": img}   
def bboxes():
    bbox_texts = glob(TRAIN_DIR + "*/*_boxes.txt")
    
    bbox_annotations = []
    for bbox_txt in bbox_texts:
        bbox_annotations.extend(_bbox(bbox_txt))
    return bbox_annotations
            
def _train():
    bbox = bboxes()    
    cls_name = class_name()
    
    # INFO [image, (label, bbox)]
    # INFO {"n02795169": "cat"} 
    # INFO -> {"fname": "n02795169_210.JPEG", "bbox":(0, 0, 1, 1), "class": }    
    for b in bbox:
        key = b["fname"].split("_")[0]
        for name_dict in cls_name:
            if key in name_dict:
                b.update({"class":name_dict.get(key)})
        print(b)
        break
         
        
           
if __name__=="__main__":
    #class_name()
    #bboxes()
    _train()
    # words = open(WORDS, "r")
    # line = search_class(words, "n04090969")    
    # class_name = line.split("\t")
    # print(class_name) 
   
    

