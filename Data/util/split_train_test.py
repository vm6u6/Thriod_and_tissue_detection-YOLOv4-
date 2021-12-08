import os, shutil
import random


def split_data(train_num, val_num,  test_num , dir_path ):
    # prepare train folder
    images_list = []
    for img in os.listdir(dir_path):
        if img.endswith(".jpg"):
            images_list.append(img.split(".")[0])
        if img.endswith(".png"):
            images_list.append(img.split(".")[0])

    random.shuffle(images_list)
    yolo_format_folder = os.path.join( dir_path )
    train_folder = os.path.join("train")
    val_folder = os.path.join("val")
    test_folder = os.path.join("test")

    if not os.path.isdir("train"):
        os.mkdir(train_folder)    
    if not os.path.isdir("val"):
        os.mkdir(val_folder)
    if not os.path.isdir("test"):
        os.mkdir(test_folder)

    # train data 
    for train_data in images_list[:train_num]:
        
        shutil.copyfile(os.path.join(yolo_format_folder, "{}.jpg".format(train_data)),  
                        os.path.join(train_folder, "{}.jpg".format(train_data)))
            
        shutil.copyfile(os.path.join(yolo_format_folder, "{}.txt".format(train_data)),  
                        os.path.join(train_folder, "{}.txt".format(train_data)))
    
    # val_data
    print(images_list[train_num+1:train_num+1+val_num])
    for val_data in images_list[train_num+1:train_num+1+val_num]:

        shutil.copyfile(os.path.join(yolo_format_folder, "{}.jpg".format(val_data )),  
                        os.path.join(val_folder, "{}.jpg".format(val_data )))
                
        shutil.copyfile(os.path.join(yolo_format_folder, "{}.txt".format(val_data )),  
                        os.path.join(val_folder, "{}.txt".format(val_data )))
    
    # test data
    for test_data in images_list[train_num+1+val_num+1:]:
        shutil.copyfile(os.path.join(yolo_format_folder, "{}.jpg".format(test_data )),  
                        os.path.join(test_folder, "{}.jpg".format(test_data )))

        shutil.copyfile(os.path.join(yolo_format_folder, "{}.txt".format(test_data )),  
                        os.path.join(test_folder, "{}.txt".format(test_data )))

    # show total data 

    print("="*35)
    print("number of training set :", len(os.listdir(train_folder)))
    print("number of val set :", len(os.listdir(val_folder)))
    print("number of test set :", len(os.listdir(test_folder)))
    print("="*35)

if __name__ == "__main__":
    # train 80% val 10% test 10%
    # total number = 2315
    train_num = 1852
    val_num = 232
    test_num = 231

    # change path to the folder which contain the yolo txt and image.
    dir_path = "/home/amc/yolo/yolo_all/"

