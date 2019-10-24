
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys
import math
#from pykml import parser
from fastkml import kml
import xml.etree.ElementTree as ET
import numpy as np
import geopy
from geopy.distance import VincentyDistance
import re
from PIL import Image, ImageFont, ImageDraw, ImageEnhance


k = kml.KML()

w_frame = 13.64173
h_frame = 20
def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km
    radius = radius*1000*1000 # mm

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

# ================================================= Load KML data ======================================================

def xml_to_csv(path):

    xml_list = []
    for subdir, dirs, files in os.walk(path):
        for xml_file in glob.glob(subdir + '/*.kml'):
            frame_coords = "Not necessary"
            frame_step = -20.0

            #print(xml_file)
            tree = ET.parse(xml_file)
            root = tree.getroot()

            lineStrings = tree.findall('.//{http://www.opengis.net/kml/2.2}LineString')
            Placemarks = tree.findall('.//{http://www.opengis.net/kml/2.2}Placemark')
            #print(len(lineStrings))
            #print(len(Placemarks))

            for placemark, LineString in zip(Placemarks,lineStrings):
                for placemark_features in placemark:
                    if placemark_features.tag == '{http://www.opengis.net/kml/2.2}name':
                        distress_name = placemark_features.text
                    if placemark_features.tag == '{http://www.opengis.net/kml/2.2}description':
                        severity = placemark_features.text
                        severity = int(re.findall(r': (.*)\n',severity)[0])+1
                        #print(int(re.findall(r': (.*)\n',severity)[0])+1)
                        #sys.exit()
                #print(LineString)
                for LineString_feature in LineString:
                    #print(LineString_feature.text)
                    distress_coords=list()
                    if LineString_feature.tag == '{http://www.opengis.net/kml/2.2}coordinates':
                        #print(LineString_feature.tag, LineString_feature.text)
                        coords_list = LineString_feature.text.replace("\n",',').split(',')[1:-1]
                        #print(coords_list)
                        i=0
                        while i<len(coords_list):
                            distress_coords.append((float(coords_list[i+1]),float(coords_list[i])))
                            i+=2
                        if distress_name=="Weathering_MTC":
                            frame_coords = distress_coords
                            frame_step += 20.0

                    folder_name = subdir.split("\\")[-1]
                    rsp_file = xml_file.split('\\')[-1].split('.')[0]



                    #sys.exit()
                    #print(subdir)

                    size_array=(13.64173,20,3)  # in ft
                    segmented_val = 0
                    section_name = xml_file.split("\\")[-1].split(".")[0]
                    image_name = rsp_file + '      '  +str(frame_step)+ '_Image3D'
                    value = [subdir, folder_name, section_name,distress_name,severity, size_array,segmented_val,distress_coords,frame_coords,image_name]
                xml_list.append(value)
                #print(rsp_file + '      ' + str(frame_step) + '_Image3D')
        # create image name for the Weathering distress

        # Frame coordinates
        for i in range(len(xml_list)):
            # find the weathering (frame coordinates) right before each patching in the dataset.
            if xml_list[i][3] == 'Patching and Utility Cuts_MTC':
                j = i - 1
                while xml_list[j][3] != 'Weathering_MTC':
                    j -= 1
                xml_list[i][8]=xml_list[j][7]
                #print(xml_list[i])

        column_name = ['Directory', 'FolderName','SectionName','Distress','Severity','SizeArray','Segmented', 'DistressCoords','FrameCoords','ImageName']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df



image_path = os.path.join(os.getcwd(), 'DE_annotations_byKML')
kml_data = xml_to_csv(image_path)
#print('Successfully converted xml to csv.')
kml_data = kml_data.loc[ kml_data['Distress'] == 'Patching and Utility Cuts_MTC']

print(kml_data[['Distress','FrameCoords','ImageName']])
#print(kml_data['FrameCoords'])
#sys.exit()
kml_data.to_csv('training_from_KML.csv', index=None)

print("Size of KML file: {}".format(kml_data.shape))
def calc_bearing2(pointA, pointB):
    lat1 = math.radians(pointA[1])
    lat2 = math.radians(pointB[1])
    lon1 = math.radians(pointA[0])
    lon2 = math.radians(pointB[0])
    d_phi = math.log(math.tan(lat2/2+math.pi/4) / math.tan(lat1/2 +math.pi/4))
    d_lon = abs(lon1-lon2)
    if d_lon > 2*math.pi:
        d_lon = math.radians(math.degrees(d_lon)%180)
    brng = math.atan2(d_lon,d_phi)
    return math.degrees(brng)

def calc_bearing(pointA, pointB):
    lat1 = math.radians(pointA[1])
    lat2 = math.radians(pointB[1])
    lon1 = math.radians(pointA[0])
    lon2 = math.radians(pointB[0])

    y = math.sin(lon2-lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(lon2-lon1)
    bearing = math.degrees(math.atan2(y, x))
    return bearing
def calc_angle(frame_coord_left,distress_coord,frame_coord_right):
    """Return the angle between two vectors in any dimension space,
    in degrees."""
    # The points in tuple latitude/longitude degrees space
    A = frame_coord_left
    B = distress_coord
    C = frame_coord_right

    # Convert the points to numpy latitude/longitude radians space
    a = np.radians(np.array(A))
    b = np.radians(np.array(B))
    c = np.radians(np.array(C))

    # Vectors in latitude/longitude space
    avec = a - b
    cvec = c - b

    # Adjust vectors for changed longitude scale at given latitude into 2D space
    lat = b[0]
    avec[1] *= math.cos(lat)
    cvec[1] *= math.cos(lat)

    # Find the angle between the vectors in 2D space
    return np.degrees(
        math.acos(np.dot(avec, cvec) / (np.linalg.norm(avec) * np.linalg.norm(cvec))))



def find_frame_corners(coords):
    bottom_left = coords[0]
    bottom_right = coords[1]
    top_left = coords[3]
    top_right = coords[2]
    return bottom_left,bottom_right,top_left,top_right

def find_distress_corners(coords):
    bottom_left = coords[3]
    bottom_right = coords[2]
    top_left = coords[0]
    top_right = coords[1]
    return bottom_left,bottom_right,top_left,top_right

def convert_coord(frame_coord_left,distress_coord,frame_coord_right,w,h):
    theta_1 = calc_bearing(frame_coord_left, frame_coord_right)
    theta_2 = calc_bearing(frame_coord_left, distress_coord)
    theta = theta_1 - theta_2
    print(calc_bearing2(frame_coord_left, frame_coord_right))
    print(calc_bearing2(frame_coord_left, distress_coord))

    sys.exit()
    print(calc_angle(frame_coord_left,distress_coord,frame_coord_right))
    print(theta_1)
    print(theta_2)
    print(theta)   # Should be 20.44
    distance = geopy.distance.distance(frame_coord_left, distress_coord).feet
    print(distance)
    sys.exit()
    x = abs(distance * math.cos(theta))
    y = abs(distance * math.sin(theta))
    return (x,y)



def convert_coords(distress_coords,frame_coords):
    frame_BL, frame_BR, frame_TL, frame_TR  = find_frame_corners(frame_coords[:-1])
    distress_BL, distress_BR, distress_TL, distress_TR = find_distress_corners(distress_coords[:-1])
    w = geopy.distance.distance(frame_BL, frame_BR).feet
    h = geopy.distance.distance(frame_BL, frame_TL).feet
    print(find_frame_corners(frame_coords[:-1]))
    print(find_distress_corners(distress_coords[:-1]))


    #print(calc_angle(frame_BL, distress_BL, frame_BR))
    #sys.exit()
    BL = convert_coord(frame_BL, distress_BL, frame_BR,w, h)
    BR = convert_coord(frame_BL, distress_BR, frame_BR,w, h)
    TL = convert_coord(frame_BL, distress_TL, frame_BR,w, h)
    TR = convert_coord(frame_BL, distress_TR, frame_BR,w, h)
    print([BL,BR, TL, TR])
    sys.exit()
    return [BL,BR, TL, TR]
frame_height = 20
frame_width = 13.64173
converted_coords = list()

for i in range(kml_data.shape[0]):
    #print(kml_data.iloc[i, 7])
    #print(kml_data.iloc[i, 8])
    #sys.exit()
    print(convert_coords(kml_data.iloc[i,7],kml_data.iloc[i,8]))
    #print(find_frame_corners(kml_data.iloc[i,7]))
    #print(find_frame_corners(kml_data.iloc[i, 8]))
    sys.exit()


def draw_box(data):
    for i in range(data.shape[0]):
        convert_coords()
        exit()
#print(kml_data[['ImageName']])

#data.to_csv('data_merged_before.csv',index=False)
#print(data[['ImageName']])
#sys.exit()

#data.to_csv('data_merged_after.csv',index=False)