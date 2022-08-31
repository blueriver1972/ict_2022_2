import os
import random
import xml.etree.ElementTree as ET
from glob import glob

import cv2
from utils import BBox

ROOT_DIR = "../data/VOC2012/"

Bbox = tuple[tuple[int, int], tuple[int, int]]
Img = list[list[list[int]]]
# pixel = list[r, g, b]
# row = list[pixel]
# col = list[row]


class VOC:
    def __init__(self, root_dir=None):
        self.root_dir = root_dir
        self.annotations_dir = root_dir + "Annotations/"
        self.img_dir = root_dir + "JPEGImages/"

    def img_n_bbox(self, xml_name) -> tuple[Img, list[Bbox]]:
        xml = self.get_xml(xml_name)

        #
        img_path = voc.img_dir + xml.findtext("filename")
        print(img_path)
        img = cv2.imread(img_path)

        #
        bboxs = []
        for obj in xml.findall("object"):
            bbox = obj.find("bndbox")
            start = int(bbox.findtext("xmin")), int(bbox.findtext("ymin"))
            end = int(bbox.findtext("xmax")), int(bbox.findtext("ymax"))
            bboxs.append((start, end))

        return img, bboxs

    def get_xml(self, xml_name):
        tree = ET.parse(self.annotations_dir + xml_name)
        return tree.getroot()

    def bbox_example(self, nums=10, shufle=True):
        bbox = BBox()
        globs = glob(self.annotations_dir + "*.xml")

        for _ in range(nums):
            fname = random.choice(globs)
            fname = os.path.basename(fname)
            img, bboxs = voc.img_n_bbox(fname)

            bbox.show(img, bboxs)


if __name__ == "__main__":
    voc = VOC(ROOT_DIR)
    voc.bbox_example()
