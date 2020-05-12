import os     
import cv2
import dlib   
from imutils import face_utils

labels=['Diamond','Oblong','Oval','Round','Square','Triangle']

train_data=[]       #empty list for saving features
train_target=[]     #empty list for saving labels

face_detector=dlib.get_frontal_face_detector()
landmark_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def create_features(points,label):

    label_dict={'Diamond':0,'Oblong':1,'Oval':2,'Round':3,'Square':4,'Triangle':5}
    
    my_points=points[2:9,0]

    D1=my_points[6]-my_points[0]
    D2=my_points[6]-my_points[1]
    D3=my_points[6]-my_points[2]
    D4=my_points[6]-my_points[3]
    D5=my_points[6]-my_points[4]
    D6=my_points[6]-my_points[5]

    d1=D2/float(D1)*100
    d2=D3/float(D1)*100
    d3=D4/float(D1)*100
    d4=D5/float(D1)*100
    d5=D6/float(D1)*100

    train_data.append([d1,d2,d3,d4,d5])
    train_target.append(label_dict[label])

for label in labels:

    path=os.path.join('Face Shapes',label)
    img_names=os.listdir(path)

    for img_name in img_names:

        img_path=os.path.join(path,img_name)
        img=cv2.imread(img_path)

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        rect=face_detector(gray)
        points=landmark_detector(gray,rect[0])
        points=face_utils.shape_to_np(points)

        create_features(points,label)

        #cv2.imshow('LIVE',img)
        #cv2.waitKey(100)

import pickle
import numpy as np

train_data=np.array(train_data)    #converting python lists into numpy arrays
train_target=np.array(train_target)

pickle.dump(train_data,open('data.pickle','wb'))
#pickle dump- for saving arrays, wb-write in bytes mode
pickle.dump(train_target,open('labels.pickle','wb'))


