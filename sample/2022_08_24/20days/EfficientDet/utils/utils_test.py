import cv2
from utils import BBox


DIR = "../data/VOC2012/JPEGimages/2007_000027.jpg" 


if __name__=="__main__":
    bbox = BBox()
    
    img = cv2.imread(DIR)
    #cv2.imshow("aa", img)
    cv2.waitKey(0)
    bbox.show(img, ((174, 101), (349, 351)))