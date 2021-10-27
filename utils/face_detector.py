import numpy as np
import cv2
import time
from mtcnn import MTCNN

detector = MTCNN()

def process_image(image, ratio=0.752, min_confidence=0.9, debug=False):
    """
    Process a single image.
    Parameters
    ----------
    image : 3d numpy array
        Input image in BGR-space.
    output_directory : string
        Path to the directory where the processed images will be saved.
    detector : ?
        Passed as argument to avoid overhead.
    """
     
    timing = time.time()
    results = detector.detect_faces(image)
    timing = time.time() - timing

    image_total = np.copy(image)    # Avoid blue frames when saving cropped faces.
    
    n_faces = len(results)

    faces = []
    
    if debug:
        print("========================")
        print(f"{image.shape=}")
        print(f"{timing=}")
        print(f"{n_faces=}")
        print(f"image size: {image.size*image.itemsize*1e-6:.1f} MB")

    if not results:
        if(debug):
            print(f"No faces detected in image. Skipping.")

    for face_idx, result in enumerate(results):
        """
        Save crop of each face.
        """
        if result['confidence'] < min_confidence:
            if debug:
                print(f"Face {face_idx + 1} of {n_faces} skipped with confidence: {result['confidence']}")
            continue
        
        if debug:
            print(f"{result['confidence']}")
        
        (x, y, width, height) = result['box']
        width_delta = int(((height * ratio) - width)/2)

        image_cropped = image[
                        y:y + height,
                        x-width_delta:x + width + width_delta, :
                    ]

        faces.append([image_cropped,result['box'], result['keypoints']])            
    

    if debug:
        print("========================\n")
    
    return image_total, faces

