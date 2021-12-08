import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
path = './yolov4_mask3_scartch/parameter_log/'
lines = 1007835   #modify here
result = pd.read_csv( path + 'train_log_iou.txt', skiprows=[x for x in range(lines) if (x%10==0 or x%10==9) ] ,
                       names=['Region Avg IOU','Count','Class_loss', 'IOU_loss','Total_loss'])


result['Region Avg IOU']=result['Region Avg IOU'].str.split('(').str.get(1)
result['Region Avg IOU']=result['Region Avg IOU'].str.split(')').str.get(0)
result['Region Avg IOU']=result['Region Avg IOU'].str.split(':').str.get(1)

result['Region Avg IOU']=pd.to_numeric(result['Region Avg IOU'])
result.dtypes

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(result['Region Avg IOU'].values,label='Region Avg IOU')

ax.legend(loc='best')
# ax.set_title('The Region Avg IOU curves')
ax.set_title('The Region Avg IOU curves')
ax.set_xlabel('batches')
# fig.savefig('Avg IOU')
fig.savefig( path + 'region avg IOU')