from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import os
import glob
import sys
import numpy as np
import pandas as pd
import re
from geopy import distance

path = os.getcwd()
image_path = os.path.join(os.getcwd(), 'Images')
dataset = pd.read_csv(os.path.join(os.getcwd(), 'dataset.csv'))
#print(dataset.head())
print(dataset.shape)
print(list(dataset.columns))
box_indx = dataset.columns.get_loc("ConvertedCoordinates")
img_indx = dataset.columns.get_loc("ImageName")
distress_indx = dataset.columns.get_loc("Distress")
frame_indx = dataset.columns.get_loc("FrameCoords")
def read_coords(box):
    # Some numbers were scientific some others were floats. e \- and \+ account for the scientific ones.
    matches = re.findall(r'\d\d?\d?.[\d?]*e?\-?\+?[\d?]*',box)
    #print(box)
    #print(matches)
    box_list = list()
    i=0
    while i <= len(matches) - 1:
        box_list.append(np.asarray([float(matches[i]),float(matches[i+1])]))
        i+=2
    return box_list

def draw_box(image_path,distresses,image_file):
    '''
    :param image_path:
    :param distresses: Containes box_coordinate,distress pairs for each frame
    :param image_file:
    :return:
    '''
    #source_img = Image.open(image_path).convert("RGBA")
    #print(os.getcwd())
    #print(image_path)
    source_img = Image.open(image_path)
    image_width = source_img.size[0]    # in pixel
    image_height = source_img.size[1]   # in pixel
    frame_height = 20                   # in ft

    for distress_set in distresses:
        box = read_coords(distress_set[0])
        distress = distress_set[1]
        frame_width = distress_set[2]  # in ft
        scale_factor = image_width / frame_width

        for i in range(len(box)):
            box[i] = box[i] * scale_factor
            box[i][1] = image_height - box[i][1]    # Beacase in PIL pixels are starrted counting from top left corner
                                                    # and I started counting the feet values from the bottom left corner
                                                    # So vertical coordinates should be reveresed but no need to change the horizontals
        draw = ImageDraw.Draw(source_img)
        draw.rectangle([box[0][0],box[0][1],box[-1][0],box[-1][1]],outline='yellow')
        draw.text(list((box[0]+box[1])/2-5), distress)
        image_name_annotated = os.getcwd()+"\\Images\\Annotated\\"+image_file+'_annotated.JPG'
        source_img.save(image_name_annotated,quality=100)
        #source_img.show()



for i in range(dataset.shape[0]):
    #print(i)
    if i==0:
        j = i
    if i == j:
        #print(i)
        #print(image_path)
        for img_path in glob.glob(image_path + "\\*.JPG"):
            image_file = img_path.split('\\')[-1][:-4]
            if dataset.iloc[i,img_indx]==image_file:
                #print("hi")
                # To account for those frames with multiple distresses
                frame_box = read_coords(dataset.iloc[i, frame_indx])
                frame_width = distance.distance(frame_box[0], frame_box[1]).feet
                distresses = [[dataset.iloc[i, box_indx],dataset.iloc[i, distress_indx],frame_width]]
                j=i+1
                try:
                    test = dataset.iloc[j,img_indx]
                except IndexError:
                    break
                while dataset.iloc[j,img_indx]==image_file:
                    #print(j)
                    frame_box = read_coords(dataset.iloc[j, frame_indx])
                    frame_width = distance.distance(frame_box[0], frame_box[1]).feet
                    distresses.append([dataset.iloc[j, box_indx],dataset.iloc[j, distress_indx],frame_width])
                    j+=1
                draw_box(img_path,distresses,image_file)
                #print("hi")
                break


