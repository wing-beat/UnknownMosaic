# video에서 unknown에 대한 처리
# USAGE
# python unknown_processing_video.py --encodings encodings.pickle --input videos/video.mp4
# python unknown_processing_video.py --encodings encodings.pickle --input videos/video.mp4 --method overlay --sticker stickers/osw.png

# import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import os
from moviepy.editor import *
import numpy as np

# overlay function
def overlay(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", type=str, help="path to output video")
ap.add_argument("-y", "--display", type=int, default=0, help="whether or not to display output frame to screen")
ap.add_argument("-d", "--method", type=str, default="mosaic", help="unknown processing method by `mosaic` or `overlay`")
ap.add_argument("-s", "--sticker", type=str, default="sticker.png", help="choice the sticker png you want `sticker_sw` or `sticker_mj`")
ap.add_argument("-m", "--sound", type=str, default="sound.mp3",help="path to sound file separated from video file")

args = vars(ap.parse_args())
m = args["method"]

video = VideoFileClip(args["input"])
video.audio.write_audiofile(args["sound"])

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the pointer to the video file and the video writer
print("[INFO] processing video...")
stream = cv2.VideoCapture(args["input"])
writer = None

# loop over frames from the video file stream
while True:
    # grab the next frame
    (grabbed, frame) = stream.read()

    # if the frame was not grabbed, break
    if not grabbed:
        break

    # convert the input frame from BGR to RGB then resize it
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    # detect each face in the input image, then compute the facial embeddings
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    # initialize the list of names for each face detected
    face_names = []

    # loop over the facial embeddings
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        distances = face_recognition.face_distance(data["encodings"], face_encoding)
        min_value = min(distances)

        name = "Unknown"
        if min_value < 0.4:
            index = np.argmin(distances)
            name = data["names"][index]

    face_names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(face_locations, face_names):
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)
        if (name == 'Unknown'):
            # draw the predicted face name on the image
            if (m == 'mosaic'):
                face_img = frame[top:bottom, left:right]
                face_img = cv2.resize(face_img, ((right-left)//30, (bottom-top)//30))
                face_img = cv2.resize(face_img, (right-left,bottom-top), interpolation=cv2.INTER_AREA)
                frame[top:bottom, left:right] = face_img
            elif (m == 'overlay'):
                sticker = cv2.imread(args["sticker"], cv2.IMREAD_UNCHANGED)
                try:
                    frame = overlay(frame, sticker, (right+left)/2, (bottom+top)/2, overlay_size=(right-left+25, bottom-top+25))
                except:
                    pass
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 24, (frame.shape[1], frame.shape[0]), True)

    # if the writer is not None, write the frame with recognized
    # faces t odisk
    if writer is not None:
        writer.write(frame)

# close the video file pointers
stream.release()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
