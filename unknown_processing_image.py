# image에서 unknown에 대한 처리
# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image testset/test.jpg --method overlay --sticker stickers/mj.png

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
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
ap.add_argument("-e", "--encodings", required=True,
   help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
   help="path to input image")
ap.add_argument("-o", "--output", help="path to save the output")
ap.add_argument("-d", "--method", type=str, default="mosaic",
   help="unknown processing method by `mosaic` or `overlay`")
ap.add_argument("-s", "--sticker", type=str, default="overlay/sticker.png",
   help="choice the sticker png you want `stickers/sw` or `stickers/mj`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect each face in the input image, then compute the facial embeddings
print("[INFO] recognizing faces...")
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
m = args["method"]
for ((top, right, bottom, left), name) in zip(face_locations, face_names):
    if (name == 'Unknown'):
        if (m == 'mosaic'):
            face_img = image[top:bottom, left:right]
            face_img = cv2.resize(face_img, ((right-left)//30, (bottom-top)//30))
            face_img = cv2.resize(face_img, (right-left,bottom-top), interpolation=cv2.INTER_AREA)
            image[top:bottom, left:right] = face_img
        elif (m == 'overlay'):
            sticker = cv2.imread(args["sticker"], cv2.IMREAD_UNCHANGED)
            image = overlay(image, sticker, (right+left)/2, (bottom+top)/2, overlay_size=(right-left+15, bottom-top+15))
        else:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

if args["output"]: cv2.imwrite(args["output"], image)

cv2.waitKey(0)
