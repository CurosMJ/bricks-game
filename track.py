import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
paramsDetector = cv2.SimpleBlobDetector_Params()
paramsDetector.filterByArea = True
paramsDetector.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(paramsDetector)

def blank():
    return None

def nothing(x):
    return None

cap = cv2.VideoCapture(0)
cv2.namedWindow('image')
cv2.namedWindow('leye')
cv2.createTrackbar('threshold', 'image', 32, 255, nothing)

leftHandler = blank
rightHandler = blank

def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        cv2.rectangle(img, (x, y), (x + w, y + h), (230, 230, 0), 2)
        frame = img[y:y + h, x:x + w]
    return frame


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)
    width = np.size(img, 1)
    height = np.size(img, 0)
    left_eye = None
    right_eye = None
    leyebw = None

    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2
        color = (255, 0, 0)
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
            leyebw = gray_frame[y:y + h, x:x + w]
            color = (0, 0, 255)
        else:
            right_eye = img[y:y + h, x:x + w]
            color = (255, 0, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)        

    if left_eye is not None:
        thresh = cv2.getTrackbarPos('threshold', 'image')
        _, processed = cv2.threshold(leyebw, thresh, 255, cv2.THRESH_BINARY)
        # processed = cv2.flip(processed, +1)
        cv2.imshow('leye', processed)
        h,w = processed.shape
        leftST = processed[:h, :int(w/2)]
        leftWhite = cv2.countNonZero(leftST)
        
        rightST = processed[:h, int(w/2):w]
        rightWhite = cv2.countNonZero(rightST)

        # print((leftWhite, rightWhite))
        if abs(leftWhite - rightWhite) > 50:
            if rightWhite > leftWhite:
                rightHandler()
            elif rightWhite < leftWhite:
                leftHandler()

    # if left_eye is None and right_eye is not None:
    #     rightHandler()

    # if left_eye is not None and right_eye is None:
    #     leftHandler()

    return left_eye, right_eye

def set_handlers(left, right):
    leftHandler = left
    rightHandler = right

def frame():
    _, frame = cap.read()
    # frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
    face_frame = detect_faces(frame, faceCascade)
    if face_frame is not None:
        detect_eyes(face_frame, eyeCascade)
    frame = cv2.flip(frame, +1)
    cv2.imshow('image', frame)

def destroy():
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
