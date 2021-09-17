# gods-eye

An AI IOT surveillance system.



## Four modules to rule them all:

```python
#### IMPORT
from utils import face_detector as fd, face_identifier as fi, face_analyzer as fa, draw_face_information as dfi

#### FACE DETECTION
image_total, faces = fd.process_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), debug=DEBUG)

#### FACE IDENTIFICATION
faces_df = fi.find(face)

#### FACE ANALYSIS
analyzis_object = fa.analyze_face(face)

#### DRAW INFORMATION
dfi.draw_information(image_total, loc, faces_df, analyzis_object)
```
