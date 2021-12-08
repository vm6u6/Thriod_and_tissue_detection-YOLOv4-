import pandas as pd 
import sys
import os

def each_class_num ( path ):
    # change the path to the yolo txt file
    
    cnt = [ 0, 0, 0, 0, 0, 0, 0 ]

    for f in os.listdir(path):
        if (f.endswith(".txt")):
            cnt[6] = cnt[6] + 1
            df = pd.read_csv( path + f, names=["classes"]) 
            df = pd.DataFrame(df.classes.str.split(' ',1).tolist(), columns = ['classes','position'])
            for num in df["classes"]:
                if num == "0":
                    cnt[0] = cnt[0] + 1
                if num == "1":
                    cnt[1] = cnt[1] + 1
                if num == "2":
                    cnt[2] = cnt[2] + 1
                if num == "3":
                    cnt[3] = cnt[3] + 1
                if num == "4":
                    cnt[4] = cnt[4] + 1
                if num == "5":
                    cnt[5] = cnt[5] + 1

    sys.stdout=open("TRAIN_CLASS_NUM.txt","w")        
    print( "--------Apply these numbers to tune the class weight loss in yolo.cfg's [yolo] layers--------" ) 
    print("") 
    print( "counter_per_class:", cnt[0], ",", cnt[1], ",", cnt[2], ",", cnt[3], ",", cnt[4], ",", cnt[5] )      
    print("")
    print( "nodule:", cnt[0], "trachea:", cnt[1], "strap_muscle:", cnt[2], "artery:", cnt[3], "vein:", cnt[4], "esophagus:", cnt[5] )
    print( "total_training_num:", cnt[6] )
    print("")
    sys.stdout.close()

if __name__ == "__main__":
    path = "/home/amc/yolo/train/"
    each_class_num(path)

