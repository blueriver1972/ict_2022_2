import os
import re
from glob import glob
import cv2
import random
import math


WNIDS = "./data/tiny-imagenet-200/wnids.txt"
WORDS = "./data/tiny-imagenet-200/words.txt"
TRAIN_DIR = "./data/tiny-imagenet-200/train/"


class Dataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.num_train_dataset = 200 * 500
        self.train_metadata = self._train_metadata()

    # INFO: [image, (label, bbox)]
    def search_class(self, fd=None, match=""):
        fd.seek(0)
        for line in fd:
            if re.search(match, line):
                class_name = line.strip().split("\t")[1]
                return class_name

    # INFO: {"n02795169": "cat"}
    def class_name(self):
        # {"wnids": "class_name"}
        wnids = open(WNIDS, "r")
        words = open(WORDS, "r")

        class_names = []
        while True:
            wnid = wnids.readline().strip()
            class_dict = {wnid: self.search_class(words, wnid)}
            if not wnid:
                break
            class_names.append(class_dict)

        wnids.close()
        words.close()
        return class_names

    def _bbox(self, bbox_txt) -> list:

        with open(bbox_txt, "r") as fd:
            dir_name = os.path.join(os.path.dirname(bbox_txt), "images")

            bboxes = []
            for line in fd:
                line = line.strip()
                fname, xmin, ymin, xmax, ymax = line.split("\t")
                #img = cv2.imread(os.path.join(dir_name, fname))
                fname = os.path.join(dir_name, fname)
                bbox_dict = {
                    "fname": fname,
                    "bbox": (int(xmin), int(ymin), int(xmax), int(ymax)),
                    #"image": img
                }
                bboxes.append(bbox_dict)
        return bboxes

# INFO: {"fname": "n02795169_210.JPEG", "bbox": (0, 0, 1, 1)}
# TODO: -> {"fname": "n02795169_210.JPEG", "bbox": (0, 0, 1, 1), "image": img}
    def bboxes(self):
        bbox_texts = glob(TRAIN_DIR + "*/*_boxes.txt")

        bbox_annotations = []
        for bbox_txt in bbox_texts:
            bbox_annotations.extend(self._bbox(bbox_txt))
        return bbox_annotations

    def _train_metadata(self):
        bbox = self.bboxes()
        cls_name = self.class_name()

        # INFO: [image, (label, bbox)]
        # INFO: {"fname": "n02795169_210.JPEG", "bbox": (0, 0, 1, 1), "image": img}
        # INFO: -> {"fname": "n02795169_210.JPEG", "bbox": (0, 0, 1, 1), "class": "cat"}
        # INFO: -> {"fname": "n02795169_210.JPEG", "bbox": (0, 0, 1, 1), "class": "cat", "image": img}
        new = []
        for b in bbox:
            key = b["fname"].split("_")[0]
            for name_dict in cls_name:
                if key in name_dict:
                    b.update({"class": name_dict.get(key)})
                    new.append(b)
        return new

    #TODO list comprehension
    def _images(self, metadata):
        images = {}
        for element in metadata:
            img = cv2.imread(element["fname"])
            images.append(img)
        return images
    
    def _labels(self, metadata):
        labels = []
        for element in metadata:
            bbox = (element["bbox"])
            clss = (element["class"])
            label = {"bbox": bbox, "class": clss}

        return images, labels    
        
        
    def call(self):
        self._train_metadata()
        batch_metadata = random.sample(self.train_metadata, self.batch_size)
        images =self._images(ba)
        labels =_
        return images, labels

    
if __name__ == "__main__":
    ds = Dataset(batch_size=32)
    #INFO 1000 / 32 = 31.25 -- ceil() --> 32
    for i in range(math.ceil(ds.num_train_dataset/ds.batch_size)):
        images, labels = ds()
        print(images)
        print(labels)
