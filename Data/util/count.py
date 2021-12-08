
from bs4 import BeautifulSoup
import pandas as pd
import sys
import os 
import numpy as np

def count( path ):
   cnt = [ 0, 0, 0, 0, 0, 0 ]
   d = os.listdir( path )
   for file in d:
      soup = BeautifulSoup(open( path + "/" + file, encoding='utf-8'), 'xml')
      name = soup.find_all('name')
      for i in name:
         if ( i.string == 'nodule' ):
            cnt[0] = cnt[0] + 1
         if ( i.string == 'trachea' ):
            cnt[1] = cnt[1] + 1
         if ( i.string == 'strap_muscle' ):
            cnt[2] = cnt[2] + 1
         if ( i.string == 'artery' ):
            cnt[3] = cnt[3] + 1
         if ( i.string == 'vein' ):
            cnt[4] = cnt[4] + 1
         if ( i.string == 'esophagus' ):
            cnt[5] = cnt[5] + 1

   sys.stdout=open("CLASS_NUM.txt","w") 
   print("")
   print( "nodule:", cnt[0], "trachea:", cnt[1], "strap_muscle:", cnt[2], "artery:", cnt[3], "vein:", cnt[4], "esophagus:", cnt[5] )
   sys.stdout.close()

def plusone(path):
   d = os.listdir( path )
   n = 0
   np.sort(d)
   for file in d:
      base = os.path.basename(file)
      num = os.path.splitext(base)[0]
      new = path + '/' + str(int(num) - 1)
      os.rename(path + '/' + file,new)
      print(new)

if __name__=='__main__':
   path = "./xml"
   #count( path )
   plusone( path )