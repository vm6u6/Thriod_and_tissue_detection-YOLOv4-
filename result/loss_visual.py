import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

path = './yolov4_12a/'
lines =35372

result = pd.read_csv( path + 'train_log_loss.txt' ,skiprows = 1, index_col = 0, 
					  names=['loss', 'rate', 'seconds', 'images', 'time'])
result.head()

result[['loss', 'avg']] = result['loss'].str.split(' ', 1, expand = True)

 
result['loss'] = result['loss'].str.split(' ').str.get(1)
result['avg'] = result['avg'].str.split(' ').str.get(0)
result['rate'] = result['rate'].str.split(' ').str.get(1)
result['seconds'] = result['seconds'].str.split(' ').str.get(1)
result['images'] = result['images'].str.split(' ').str.get(1)


result['loss'] = pd.to_numeric(result['loss'],errors='ignore')
result['avg'] = pd.to_numeric(result['avg'],errors='ignore')
result['rate'] = pd.to_numeric(result['rate'],errors='ignore')
result['seconds'] = pd.to_numeric(result['seconds'],errors='ignore')
result['images'] = pd.to_numeric(result['images'],errors='ignore')


fig = plt.figure(1, dpi=160)
ax = fig.add_subplot(1, 1, 1)
ax.plot(result['avg'].values,'r', label='avg loss')
ax.legend(loc='best')
#ax.set_ylim([0,1.25])
#ax.set_xlim([0, 200])
ax.set_title('The avg loss curves')
ax.set_xlabel('iteration')
ax.spines['top'].set_visible(False)     
ax.spines['right'].set_visible(False)   
fig.savefig( path + 'avg loss')

