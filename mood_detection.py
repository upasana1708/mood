import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

img_rows = 64
img_cols = 64

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# Takes an Image as Input
# Returns location & size of detected face, detected face image in gray scale (ROI) of dim 64x64 pixel and the marked input image
# ROI = Region of Interest
def face_detector(img):

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # call face classifier method with gray scale image
    # this returns a tuple (x, y, w, h)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # if no face is found, return zeros for location and size of face, zero array for roi, and original image as is
    if faces is ():
        return (0, 0, 0, 0), np.zeros((img_rows, img_cols), np.uint8), img

    # for each face found, draw a rectangle around the face on the image
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # copy the face image array to roi gray
        roi_gray = gray[y:y + h, x:x + w]

    try:
        # resize the face roi image to 64 x 64 pixel
        roi_gray = cv2.resize(roi_gray, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
    except:
        # in case of exception while resizing the roi image, just return zero array for roi
        return (x, y, w, h), np.zeros((img_rows, img_cols), np.uint8), img

    # if everything is succesful, finally return the required parameters
    return (x, y, w, h), roi_gray, img

mood_classifier = load_model('./_mini_XCEPTION.102-0.66.hdf5')
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)

    if np.sum([face]) != 0.0:

        rescaled_roi = face.astype("float") / 255.0
        roi_array = img_to_array(rescaled_roi)
        expanded_roi_array = np.expand_dims(roi_array, axis=0)

        label_scores_array = mood_classifier.predict(expanded_roi_array)
        preds = label_scores_array[0]
        label = EMOTIONS[preds.argmax()]
        label_position = (rect[0], rect[1])
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(image, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('All', image)

    if cv2.waitKey(1) == 13:  # 13 is for the Enter Key
        break

cap.release()
cv2.destroyAllWindows()


