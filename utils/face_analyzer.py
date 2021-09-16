from deepface.DeepFace import analyze



def analyze_face(image):
    return analyze(image, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)