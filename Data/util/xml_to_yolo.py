import os
import shutil
from bs4 import BeautifulSoup

def run_convert(all_classes, train_img, train_annotation, yolo_path, write_txt):
    now_path = os.getcwd()
    data_counter = 0 #change the number to different

    for data_file in os.listdir(train_annotation):
        try:
            with open(os.path.join(train_annotation, data_file), 'r', encoding="utf-8") as f:
                print("read file...")
                soup = BeautifulSoup(f.read(), 'xml')
                img_name = soup.select_one('filename').text

                for size in soup.select('size'):
                    img_w = int(size.select_one('width').text)
                    img_h = int(size.select_one('height').text)
                    
                img_info = []
                for obj in soup.select('object'):
                    xmin = int(obj.select_one('xmin').text)
                    xmax = int(obj.select_one('xmax').text)
                    ymin = int(obj.select_one('ymin').text)
                    ymax = int(obj.select_one('ymax').text)
                    objclass = all_classes.get(obj.select_one('name').text)

                    x = (xmin + (xmax-xmin)/2) * 1.0 / img_w
                    y = (ymin + (ymax-ymin)/2) * 1.0 / img_h
                    w = (xmax-xmin) * 1.0 / img_w
                    h = (ymax-ymin) * 1.0 / img_h
                    img_info.append(' '.join([str(objclass), str(x),str(y),str(w),str(h)]))

                # copy image to yolo path and rename
                img_path = os.path.join(train_img, img_name)

                # if split('.')[1] would output IMA f, split('.')[2] would output jpg
                img_format = img_name.split('.')[1]  # jpg or png
                
                shutil.copyfile(img_path, yolo_path + str(data_counter) + '.' + img_format)
                
                # create yolo bndbox txt
                with open(yolo_path + str(data_counter) + '.txt', 'a+') as f:
                    f.write('\n'.join(img_info))

                # create train or val txt
                with open(write_txt, 'a') as f:
                    path = os.path.join(now_path, yolo_path)
                    line_txt = [path + str(data_counter) + '.' + img_format, '\n']
                    f.writelines(line_txt)

                data_counter += 1
                    
        except Exception as e:
            print(e)
           
    print('the file is processed')

if __name__ == '__main__':          
    all_classes = {'nodule':0, 'trachea':1, 'strap_muscle':2, 'artery':3, 'vein':4, 'esophagus':5}
    # image file
    train_img = "/home/amc/yolo/DATA/raw_data/Transverse_2014/img"
    # the xml file
    train_annotation = "/home/amc/yolo/DATA/raw_data/Transverse_2014/yolo_label/converted"
    # the path to save the converted file,
    yolo_path = "yolo_train/"
    # save all yolo file direction in a txt file in order to feed in the darknet model.
    write_txt = 'cfg/train_tra.txt'


    if not os.path.exists(yolo_path):
        os.mkdir(yolo_path)
    '''
    else:
        lsdir = os.listdir(yolo_path)
        for name in lsdir:
            if name.endswith('.txt') or name.endswith('.jpg') or name.endswith('.png'):
                os.remove(os.path.join(yolo_path, name))
    '''
    cfg_file = write_txt.split('/')[0]
    if not os.path.exists(cfg_file):
        os.mkdir(cfg_file)
        
    if os.path.exists(write_txt):
        file=open(write_txt, 'w')

    run_convert(all_classes, train_img, train_annotation, yolo_path, write_txt)
