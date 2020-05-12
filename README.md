#*************************************************************************************************************************************
#######***Face Type Detector Python Application***########
#***************************************************************************************************************************************
Predicting the face type of a person according to the 6 standard face types


![sample](https://user-images.githubusercontent.com/64163110/81740840-38be1d00-94bb-11ea-8a00-75c59866a7f2.JPG)


****Pre-requisites****** 

  1. Python 3.6
  2. OpenCV 
  3. numpy
  4. imutils
  5. scikit-learn
  6.pickle
  7.dlib
  8.joblib


### Procedure you should follow 

  **step 1:**Meke sure  "Face Shapes" folder & "shape_predictor_68_face_landmarks.dat" is in the file location
  
  **step 2:**Run the "1.Creating the dataset.py" and make sure "data.pickle" & " label.pickle" files are created
  
  **step 3:**Run the "2.Training & Save Model.py" and cheak "Model.sav" is created
  
  **step 4:** Run the "3.face type detector.py" and find your face type
 
****Improvements can be made*****

  1. If we use more imgaes rather than  dataset acccuracy can be improved.
  2. Can use nural network instead of using svm.
  3. Can  improve to take capture few images from a face and  check those images  type and show the majority of those it will be maybe accurate
