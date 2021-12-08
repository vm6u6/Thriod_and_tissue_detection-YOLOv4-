import os
import sys
from voc_eval import voc_eval

current_path = os.getcwd()
results_path = current_path+ '/yolov4_mask3_scartch/each_class_ap_val'
sub_files = os.listdir(results_path)
sys.stdout=open( "Each_class_ap.txt","w") 
mAP = []
for i in range(len(sub_files)):
    class_name = sub_files[i].split(".txt")[0]  
    rec, prec, ap = voc_eval('/home/amc/yolo/Result/yolov4_mask3_scartch/each_class_ap_val/{}.txt', 
                            '/home/amc/yolo/Data/count/xml/{}',
                            '/home/amc/yolo/cfg/val_filename.txt',
                            class_name,
                            './yolov4_mask3_scartch')
    print("{} :\t {} ".format(class_name, ap))
    mAP.append(ap)

mAP = tuple(mAP)

print("***************************")
print("mAP :\t {}".format( float( sum(mAP)/len(mAP)) ))
sys.stdout.close()