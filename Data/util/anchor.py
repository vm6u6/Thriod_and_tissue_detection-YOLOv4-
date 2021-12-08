import numpy as np 

def anchor( out ):
    x = []
    y = []
    area = []

    for i in range(len(out)):
        x.append(out[i][0]*416)
    for i in range(len(out)):
        y.append(out[i][1]*416)

    for i in range(len(x)):
        area.append(x[i] * y[i])

    new_x = [0 for _ in range(len(x))]
    new_y = [0 for _ in range(len(y))]

    for i in range(len(np.argsort(area))):
        new_x[i] = int(x[np.argsort(area)[i]])
        new_y[i] = int(y[np.argsort(area)[i]])
        
    anchors = []
    for i in range(len(new_x)):
        anchors.append(new_x[i])
        anchors.append(new_y[i])
        
    print(" ANCHOR: ",anchors)
    print(" ")
    return anchors

