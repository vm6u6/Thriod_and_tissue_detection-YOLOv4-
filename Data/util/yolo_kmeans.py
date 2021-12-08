#coding=utf-8
import xml.etree.ElementTree as ET
import numpy as np
import os
import glob2
from glob2 import glob
from kmeans import kmeans, avg_iou
from kmeans_pp import kmeans_pp
from anchor import anchor
import sys

def load_dataset(path):
    dataset = []
    d = os.listdir( path )
    for xml_file in d:
        fullname = os.path.join(path, xml_file)
        tree = ET.parse(fullname)
        height = int(tree.findtext("./size/height"))  
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            xmin = int(obj.findtext("bndbox/xmin")) / width
            ymin = int(obj.findtext("bndbox/ymin")) / height

            xmax = int(obj.findtext("bndbox/xmax")) / width
            ymax = int(obj.findtext("bndbox/ymax")) / height
            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            if xmax == xmin or ymax == ymin:
                print(xml_file)
            dataset.append([xmax - xmin, ymax - ymin])
    return np.array(dataset)


def get_anchors( ANNOTATIONS_PATH, CLUSTERS, INPUTDIM ):
    data = load_dataset(ANNOTATIONS_PATH)
    sys.stdout=open("Kmeans_Anchors_" + str(CLUSTERS) + "_anchors" + ".txt","w")
    print('----- kmeans ------')
    out = kmeans(data, k=CLUSTERS)
    
    print('Boxes:')
    print(np.array(out)*INPUTDIM)  
    #print("Boxes:\n {}-{}".format(out[:, 0]*416, out[:, 1]*416))  
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))       
    final_anchors = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Before Sort Ratios:\n {}".format(final_anchors))
    print("After Sort Ratios:\n {}".format(sorted(final_anchors)))
    print(" ")
    print(" ")
    anchor1 = anchor(out)

    print(" ")
    print('----- kmeans++ ------')
    out2 = kmeans_pp(data, k=CLUSTERS)
    
    print('Boxes:')
    print(np.array(out2)*INPUTDIM)  
    #print("Boxes:\n {}-{}".format(out2[:, 0]*416, out2[:, 1]*416))
    print("Accuracy: {:.2f}%".format(avg_iou(data, out2) * 100))
    final_anchors2 = np.around(out2[:, 0] / out2[:, 1], decimals=2).tolist()
    print("Before Sort Ratios:\n {}".format(final_anchors2))
    print("After Sort Ratios:\n {}".format(sorted(final_anchors2)))
    print(" ")
    print(" ")
    anchor2 = anchor(out2)
    sys.stdout.close()

if __name__ == '__main__':

    ANNOTATIONS_PATH = "/home/amc/yolo/Data/count/xml" #xml files
    CLUSTERS = 6
    INPUTDIM = 608
    get_anchors( ANNOTATIONS_PATH, CLUSTERS, INPUTDIM )
    