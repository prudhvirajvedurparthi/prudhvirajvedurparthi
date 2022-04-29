import cv2
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os

cascPath = "yolomodels/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

with open('yolomodels/model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    classifier = model_from_json(loaded_model_json)
classifier.load_weights("yolomodels/model_weights.h5")
classifier._make_predict_function()


def detectSuspiciousActivity(image):
    result = 'none'
    temp = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))
    if len(faces) > 0:
        img = cv2.resize(image, (64,64))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,64,64,3)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255
        preds = classifier.predict(img)
        print(str(preds)+"   "+str(np.max(preds)))
        predict = np.argmax(preds)

        img = temp
        if predict == 1:
            result = 'Suspicious Activity Detected. Person face covered with mask'            
        if predict == 0:
            result = 'No Suspicious Activity Detected'            
    else:
        result = 'No presence of human detected'
    return result    
        
net = cv2.dnn.readNet("yolomodels/yolov3_training_2000.weights", "yolomodels/yolov3_testing.cfg")
classes = ["Crime Object Detected"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    height, width, channels = img.shape
    result = detectSuspiciousActivity(img)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    if indexes == 0: print("crime object detected in frame")
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            
    cv2.putText(img, result, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(650) & 0xFF == ord('q'):
        break   
cap.release()
cv2.destroyAllWindows()
