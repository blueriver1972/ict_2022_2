#-*- coding:utf-8 -*-
import os  
import cv2 

IMG_PATH = "../images"
IMG_SAVE_PATH = "../saveimages"

#TODO 
# - ���ο� img �Ҿ�ͼ�
# - lower�� �ѹ� �� lower�ϼ���
# - higher�� �ѹ� �� higher �ϼ����
# - imwrite�� ��� �����ϼ���

if __name__ == "__main__":
    img = cv2.imread(os.path.join(IMG_PATH, "lena.png"))
    img_logo = cv2.imread(os.path.join(IMG_PATH, "logo.png"))
    
    lower_reso = cv2.pyrDown(img)
    higher_reso = cv2.pyrUp(img)
    
    lower_logo_reso = cv2.pyrDown(img_logo)
    lower_logo_reso_1 = cv2.pyrDown(lower_logo_reso)
    
    higher_logo_reso = cv2.pyrUp(img_logo)
    higher_logo_reso_1 = cv2.pyrUp(higher_logo_reso)
       
    
    cv2.imshow('img', img)
    cv2.imshow('lower', lower_reso)
    cv2.imshow('higher', higher_reso)
  
    cv2.imwrite(os.path.join(IMG_SAVE_PATH, "lower_reso.png"), lower_reso)
    cv2.imwrite(os.path.join(IMG_SAVE_PATH, "higher_reso.png"), higher_reso)
    cv2.imwrite(os.path.join(IMG_SAVE_PATH, "lower_logo_reso.png"), lower_logo_reso)
    cv2.imwrite(os.path.join(IMG_SAVE_PATH, "lower_logo_reso_1.png"), lower_logo_reso_1)
    cv2.imwrite(os.path.join(IMG_SAVE_PATH, "higher_logo_reso.png"), higher_logo_reso)
    cv2.imwrite(os.path.join(IMG_SAVE_PATH, "higher_logo_reso_1.png"), higher_logo_reso_1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    