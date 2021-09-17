
import numpy as np
import cv2
import pandas as pd

male = pd.read_csv("data/names/herrenavn.csv")
female = pd.read_csv("data/names/kvinnenavn.csv")
maleLength = len(male['Navn'])
feMaleLength = len(female['Navn'])

def draw_information(image_total, loc, faces_df, analyzis_object, useRandomNames=False):
    currentDf = male if analyzis_object['gender'] == "Man"  else female
    currentLength = maleLength if analyzis_object['gender'] == "Man"  else feMaleLength
    randomInt = np.random.randint(1, currentLength)
    
    if (useRandomNames): 
        score = 0
        identity =  currentDf.iloc[randomInt]['Navn']
    else: 
        if(len(faces_df['identity']) > 0):
            identity = faces_df.iloc[0]['identity'].split("/")[2]
            score = faces_df.iloc[0]['VGG-Face_cosine']
        else:
            #choose random name if none identities was found
            identity =  currentDf.iloc[randomInt]['Navn']

    x, y, width, _ = loc   # x, y is the coordinate of the top left corner.

    draw_rectangle_with_opacity(image_total, loc)
    
    add_text(
        image_total, 
        text=f"Name: {identity} ({score:.2f})", 
        org=(x + width + 10, y + 25)
    )
    add_text(
        image_total, 
        text = f"Age: {analyzis_object['age']}",
        org = (x + width + 10, y + 75), 
    )
    add_text(
        image_total, 
        text = f"Sex: {analyzis_object['gender']}",
        org = (x + width + 10, y + 125), 
    )
    add_text(
        image_total, 
        text = f"Emotion: {analyzis_object['dominant_emotion']}",
        org = (x + width + 10, y + 175), 
    )
    add_text(
        image_total, 
        text = f"Race: {analyzis_object['dominant_race']}",
        org = (x + width + 10, y + 225), 
    )


def add_text(image_total, text, org):
    cv2.putText(
        img = image_total,
        text = text,
        org = org, 
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.75,
        thickness = 2,
        color = 0
    )  

def draw_rectangle_with_opacity(image_total, loc):
    x, y, width, height = loc   # x, y is the coordinate of the top left corner.
    sub_img = image_total[y:y+height, x+width:x+width+300]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    image_total[y:y+height, x+width:x+width+300] = res