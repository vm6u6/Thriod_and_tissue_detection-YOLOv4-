from bs4 import BeautifulSoup
import os 

def trade_spider( dir_path, save_path):
   d = os.listdir( dir_path )
   d = sorted(d)
   #depend on the last file number
   for file in d:
      print(file)
      soup = BeautifulSoup(open(dir_path + file, 'r',encoding='utf-8'), 'xml')
      result = soup.find('name')

      if ( result.string == 'module' ):
         result.string.replace_with('nodule')
      
      create_name = os.path.join(save_path, str(file))
      with open(create_name, 'w', encoding='utf-8') as f:
         f.write(str(soup))

if __name__=='__main__':
   dir_path = "/home/amc/yolo/DATA/raw_data/Transverse_2014/yolo_label/"
   save_path = "/home/amc/yolo/DATA/raw_data/Transverse_2014/yolo_label/converted/"
   trade_spider( dir_path, save_path)
