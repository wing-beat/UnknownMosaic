# webcam에 등장하는 사람의 얼굴이 인식되면 얼굴 부분만 crop해서 저장
# USAGE
# python build_dataset_CaffeCNN.py --output dataset/seowon

# import libraries
import os   
import cv2
import numpy
import argparse
from imutils import paths

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory of face images")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Control the camera resolution with CAM
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# get the path of the directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# create the output folder if it does not exist
if not os.path.exists(args["output"]): 
  os.makedirs(args["output"])

# import the models provided in the OpenCV repository
model1 = cv2.dnn.readNetFromCaffe('./PretrainedModel/deploy.prototxt', './PretrainedModel/weights.caffemodel')

#-- Threshold of confidence level = Th
Th = 0.65

# loop through all the files in the folder
count = 0
while True:
    # load live image data
    ret, image = cap.read()

    # accessing the image.shape tuple and taking the first two elements which are height and width
    (h, w) = image.shape[:2]

    # get our blob which is our input image after mean subtraction, normalizing, and channel swapping
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # input the blob into the model and get back the detections from the page using model.forward()
    model1.setInput(blob)
    detections = model1.forward()

    # Iterate over all of the faces detected and extract their start and end points

    for i in range(0, detections.shape[2]):
      box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")

      confidence = detections[0, 0, i, 2]

      # if the algorithm is more than Th (% )confident that the detection is a face, show a rectangle around it
      if (confidence > Th):
        # croping the image
        face_image = image[startY+2:endY-2, startX+2:endX-2]
        
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        # save the detected image to the Output folder
        cv2.imwrite(args["output"] + str(count)+'.jpg', face_image)
        count = count + 1
        
    # show the result image
    cv2.namedWindow("Detection result", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("Detection result",image)
    
    if count > 1000:
        break
    if cv2.waitKey(3) & 0xFF == ord('q'): #press q to quit
        break

cv2.destroyAllWindows()

#--- For referencing
# https://towardsdatascience.com/detecting-faces-with-python-and-opencv-face-detection-neural-network-f72890ae531c
