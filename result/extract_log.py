# coding=utf-8

import inspect
import os
import random
import sys

#log file path
log_path = "/home/amc/yolo/Result/yolov4_12a/train_yolov4.log"
save_path = "/home/amc/yolo/Result/yolov4_12a/"

def extract_log(log_file,new_log_file,key_word):
    with open(log_file, 'r') as f:
      with open(new_log_file, 'w') as train_log:
  #f = open(log_file)
    #train_log = open(new_log_file, 'w')
        for line in f:
          if 'Syncing' in line:
            continue
          if 'nan' in line:
            continue
          if key_word in line:
            train_log.write(line)
    f.close()
    train_log.close()
 
extract_log(log_path,save_path + 'train_log_loss.txt','images')
extract_log(log_path,save_path + 'train_log_iou.txt','IOU')
extract_log(log_path,save_path + 'train_log_map.txt','mAP@0.50')
extract_log(log_path,save_path + 'train_log_F1.txt','F1-score')
