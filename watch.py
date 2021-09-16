import cv2
from utils import face_detector, tests

def main():
    tests.print_computing_device()

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

        key = cv2.waitKey(1)
        if key == 27: # exit on ESC
            break
        
    cv2.destroyAllWindows()
    vc.release()

if __name__ == "__main__":
    main()