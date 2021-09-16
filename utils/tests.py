import cv2
import tensorflow as tf
from . import face_detector

def webcam_test():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        
        cv2.imshow("preview", frame)

        key = cv2.waitKey(0)
        if key == 27: # exit on ESC
            break
        
    cv2.destroyAllWindows()
    vc.release()


def face_detector_test():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()

        image_total_rgb, faces = face_detector.process_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), debug=False)
        image_total_bgr = cv2.cvtColor(image_total_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow("preview", image_total_bgr)

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
        
    cv2.destroyAllWindows()
    vc.release()

def print_computing_device():
    print("- tf computing device:")
    print(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)))
    print("-")