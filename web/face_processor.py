# coding: utf-8

# In[1]:


import sys
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from train.model import P_Net,R_Net,O_Net
import cv2
import os
import numpy as np
import train.config as config


# In[ ]:


test_mode=config.test_mode
thresh=config.thresh
min_face_size=config.min_face
stride=config.stride
detectors=[None,None,None]
# 模型放置位置
model_path=['model/PNet/','model/RNet/','model/ONet']
batch_size=config.batches
PNet=FcnDetector(P_Net,model_path[0])
detectors[0]=PNet


if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet


if test_mode == "ONet":
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet


def detection(imgle_file):

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh)
    out_path=config.out_path

    img=cv2.imread(imgle_file)
    boxes_c,landmarks=mtcnn_detector.detect(img)
    for i in range(boxes_c.shape[0]):
        bbox=boxes_c[i,:4]
        score=boxes_c[i,4]
        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        #画人脸框
        cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                      (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
        #判别为人脸的置信度
        cv2.putText(img, '{:.2f}'.format(score),
                   (corpbbox[0], corpbbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
    #画关键点
    for i in range(landmarks.shape[0]):
        for j in range(len(landmarks[i])//2):
            cv2.circle(img, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))
    #cv2.imshow('im',img)
    k = cv2.waitKey(0) & 0xFF
    #if k == 27:
    #    cv2.imwrite(out_path + item,img)
    cv2.imwrite(out_path + 'predicted_image.png',img)
    cv2.destroyAllWindows()


