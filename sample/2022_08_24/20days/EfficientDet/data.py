
import re 
from glob import glob 

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

def __bbox(fd=None) -> list:
    bboxes = []
    for line in fd:
        line = line.strip()
        fname, xmin, ymin, xmax, ymax = line.split("\t")
        bbox_dict = {
            "fname": fname, 
            "bbox": ( int(xmin), int(ymin), int(xmax), int(ymax))
            #"bbox": ( (xmin), (ymin), (xmax), (ymax))
        }
        bboxes.append(bbox_dict)
    #print(bboxes)
    return bboxes


# INFO {"fname": "n02795169_210.JPEG", "bbox":(0, 0, 1, 1)}    
def bboxes():
    bbox_texts = glob(TRAIN_DIR+"*/*_boxes.txt")
    
    bbox_annotations = []
    for bbox_txt in bbox_texts:
        fd = open(bbox_txt, "r")
        bbox_annotations.append(__bbox(fd))
        fd.close()
    return bbox_annotations
            
def _train():
    bbox = bboxes()    
    cls_name = class_name()
    
    # INFO [image, (label, bbox)]
    # INFO {"n02795169": "cat"} 
    # INFO -> {"fname": "n02795169_210.JPEG", "bbox":(0, 0, 1, 1), "class": }    
    for b in bbox:
        print(b[0])
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
   
    

