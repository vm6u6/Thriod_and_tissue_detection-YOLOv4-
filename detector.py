import cv2
import os
import numpy as np
import glob
import random
import time
import sys

weights = "/home/amc/yolo/Result/yolov4_csp/yolov4_customcsp_ema.weights"
config_file = "/home/amc/yolo/darknet/cfg/yolov4_customcsp.cfg"
save_path = "./Result/yolov4_csp/test_img/"
img_path = "./test/*.jpg"
label_path = "./Data/all_classes.names"
CONFIDENCE_THRESHOLD = 0.51 #IOU threshold
yolo_input_size = 608 #416

# Load Yolo
print('LOADING YOLO...')
net = cv2.dnn.readNet( weights, config_file )

# determine only the *output* layer names that we need from YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("YOLO LOADED!")

# Load label names
classes=[]
with open(label_path, "r") as f:
  classes = [line.strip() for line in f.readlines()]
print(classes)

images_path = glob.glob(img_path)
random.shuffle(images_path)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# loop through all the images
z = 0
for img_path in images_path:
    # Loading image
    sys.stdout=open( save_path + str(z) + '.txt',"w") 
    img = cv2.imread(img_path)
    print("-"*50)
    print("[IMAGE] ", img_path)
    height, width, channels = img.shape # 500 * 500 * 3

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (yolo_input_size, yolo_input_size), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    start = time.time()
    outs = net.forward(output_layers)
    end = time.time()
    print("[INFO] took {:.6f} seconds".format(end - start))


    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    max = 0
    for out in outs:
        for detection in out:
            
            scores = detection[5:]
            class_id = np.argmax(scores)
            #print( "CLASS_ID:", class_id )
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #print("box:", len(boxes))
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = [int(c) for c in colors[class_ids[i]]]
            text = "{}: {:.4f}".format(label, confidences[i])
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, text, (x, y -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print( label + " Area: ",round((w*h)/(500*500)*100), "%") 
            #for j in len(int(i)):
                #cv2.putText(img, "Area: "+str((x*y)/(500*500)),(int(10), int(20*j)),0, 0.5, (0,255,0),1)
    sys.stdout.close()
    # display the image       
    #cv2.namedWindow("Image",0)
    #cv2.resizeWindow("Image", 400, 400)
    #cv2.imshow("Image", img)
    #key = cv2.waitKey(0)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, str(z)+'.jpg'), img)
    cv2.waitKey(0)
    z = z + 1
    
cv2.destroyAllWindows()           
