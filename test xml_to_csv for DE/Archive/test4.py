
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
                            distress_coords.append((float(coords_list[i]),float(coords_list[i+1])))
                            i+=2
                        #print(distress_coords)
                        #sys.exit()
                    #print(distress_name)
                    folder_name = subdir.split("\\")[-1]
                    img_name = xml_file
                    #print(img_name)
                    #print(subdir)
                    size_array=(13.64173,20,3)  # in ft
                    segmented_val = 0
                    section_name = xml_file.split("\\")[-1].split(".")[0]
                    value = (subdir, folder_name, section_name,distress_name,severity, size_array,segmented_val,distress_coords)

                xml_list.append(value)
        column_name = ['Distress Name', 'Distress Coordinates']
        column_name = ['Directory', 'Folder name','SectionName','Distress','Severity','Size array','Segmented', 'Distress coordiantes']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df



image_path = os.path.join(os.getcwd(), 'DE_annotations_byKML')
kml_data = xml_to_csv(image_path)
kml_data.to_csv('training_from_KML.csv', index=None)
#print('Successfully converted xml to csv.')
kml_data = kml_data[kml_data['Distress']=='Patching and Utility Cuts_MTC']
print("Size of KML file: {}".format(kml_data.shape))

# ================================================= Load DFF data ======================================================
path = os.path.join(os.getcwd(), 'DE_annotations_byKML')
#print(image_path)
#sys.exit()
def modify_distress_name_per_line(line):
    line = line.replace(' and ', 'and')
    line = line.replace('&','and')
    line = line.replace('T C', 'TC')
    line = line.replace('Utility Cuts','UtilityCuts')
    line = line.replace('Alligator Cracking','AlligatorCracking')
    line = line.replace('Block Cracking', 'BlockCracking')
    line = line.replace('\t',' ')
    line = line.replace('\n', ' ')
    line = line.replace(' (','(')

    return line
def filter_blank_out(my_list):
    my_list_filtered = []
    for item in my_list:
        if item!='':
            item = item.replace('\t','')
            item = item.replace('\n', '')
            my_list_filtered.append(item)
    return my_list_filtered

def load_data(file_path):
    input_lines = []
    with open(file_path,'r') as f:
        line_num=1
        for line in f:
            if line_num == 1:
                rsp_file=line.split('\\')[-1].split('.')[0]
            if line_num<=2:
                line_num+=1
                continue
            line_modified = modify_distress_name_per_line(line)
            line_filtered_modified = filter_blank_out(line_modified.split(" "))
            #print(line_modified.split(" "))
            #print(filter_blank_out(line_modified.split(" ")))
            #sys.exit()
            input_lines.append(line_filtered_modified)
    dff_np=np.asarray(input_lines)
    return rsp_file, dff_np
for dff_path in glob.glob(path + '/*.dff'):  # Frame information
    print(dff_path)
    #print(type(xml_file))
    #sys.exit()
    dff_np = load_data(dff_path)[1]
    RSP_name = load_data(dff_path)[0]
    # Load Pandas Dataframes for ddf and dff files
    dff_data = pd.DataFrame(dff_np[1:,:],columns=dff_np[0,:],index=None)

    # Preparations for Image file names in the HDC folder using DFF file
    dff_data = dff_data.astype({'From_Station': 'float32', 'To_Station': 'float32'})
    dff_data[['From_Station', 'To_Station']] = dff_data[['From_Station', 'To_Station']] / 20
    dff_data = dff_data.astype({'From_Station': 'int32', 'To_Station': 'int32'})
    dff_data[['From_Station', 'To_Station']] = dff_data[['From_Station', 'To_Station']] * 20.0
    # print(ddf_data[['From(ft)','To(ft)']])
    dff_data = dff_data.astype({'From_Station': 'str', 'To_Station': 'str'})
    dff_data['ImageName'] = RSP_name + '      ' + dff_data[['From_Station']] + '_Image3D'
    dff_data['SectionName'] = RSP_name
    # print(dff_data['ImageName'])
    try:
        dff_all = dff_all.append(dff_data, ignore_index=True)
        #print(dff_data.shape)
        #print(dff_all.shape)
    except NameError:
        dff_all=dff_data
        #print(dff_data.shape)
print("Size of DFF file: {}".format(dff_all.shape))
dff_data=dff_all
#print(dff_data.shape)

# ================================================= Merge KML and DFF data =============================================
kml_data.columns = ['Directory', 'FolderName', 'SectionName', 'Distress', 'Severity',
       'Size array', 'Segmented', 'Distress coordiantes']
#print(kml_data.loc[0,'Directory'])
#print(kml_data.loc[0,'Directory'])
#print(list(kml_data.columns))
#print(list(dff_data.columns))
#sys.exit()
data = pd.merge(kml_data[['Directory', 'FolderName', 'SectionName', 'Distress', 'Severity','Size array', 'Segmented', 'Distress coordiantes']],
                 dff_data[['ImageName', 'SectionName', 'From_GPS_Lon', 'From_GPS_Lat', 'To_GPS_Lon', 'To_GPS_Lat']],
                 how='right',
                 on='SectionName')
#print(data.head())
print("KML and DFF are merged into one unique dataframe, called data (size : {}) "
      "and the following selected columns:\n{}"
      .format(data.shape,list(data.columns)))
#print(list(data.iloc[0,:]))
# ========================================= create Image name for each distress=========================================
def calc_bearing(pointA, pointB):
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    lon1 = math.radians(pointA[1])
    lon2 = math.radians(pointB[1])

    y = math.sin(lon2-lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(lon2-lon1)
    bearing = math.degrees(math.atan2(y, x))
    return bearing

def find_frame_corners(start,end):
    bearing=calc_bearing(start,end)
    bearing_perpendicular_left = bearing - 90
    bearing_perpendicular_right = bearing + 90
    # Define a general distance object, initialized with a distance of 1 ft wrt km.
    #d = geopy.distance.VincentyDistance(kilometers=0.0003048)
    d = geopy.distance.geodesic(kilometers=0.0003048)
    # Use the `destination` method with a bearing of 0 degrees (which is north)
    # in order to go from point `start` 1 km to north.
    bottom_left = d.destination(point=start, bearing=bearing_perpendicular_left)
    bottom_left = (bottom_left.latitude,bottom_left.longitude)

    bottom_right = d.destination(point=start, bearing=bearing_perpendicular_right)
    bottom_right = (bottom_right.latitude,bottom_right.longitude)

    top_left = d.destination(point=end, bearing=bearing_perpendicular_left)
    top_left = (top_left.latitude,top_left.longitude)

    top_right = d.destination(point=end, bearing=bearing_perpendicular_right)
    top_right = (top_right.latitude,top_right.longitude)

    return bottom_left,bottom_right,top_left,top_right
    #bottom_left = geopy.destination(start, bearing_perpendicular_left)
    #print(bottom_left)


def is_inside(point,start,end):
    frame_height = 20
    frame_width = 13.64173
    max_distance = math.sqrt((frame_height**2+(frame_width/2)**2))+frame_width/2 # close to 27 ft
    # Method 3 compare the SUM distance of start and end to the point and the max_distance

    if geopy.distance.distance(point,start).feet+geopy.distance.distance(point,end).feet <= max_distance:
        #print(geopy.distance.distance(point,start).feet+geopy.distance.distance(point,end).feet)
        return True
    else:
        return False



# ================================================ Merge KML and DFF data without MERGING!==============================
kml_data.columns = ['Directory', 'FolderName', 'SectionName', 'Distress', 'Severity',
       'Size array', 'Segmented', 'Distress coordiantes']
# Add dff needed columns
kml_data['bottom_left']=""
kml_data['bottom_right']=""
kml_data['top_left']=""
kml_data['top_right']=""
kml_data['ImageName']=""
print("DFF data columns:\t{}".format(list(dff_data.columns)))
print("KML data columns:\t{}".format(list(kml_data.columns)))

def find_its_image(coords,data):
    inside_count = 0
    distress_found = False
    #print(coords)
    # start with one coord and find the first match. Then check the other coords and make sure they are inside the frame range as well.
    check_coord = coords[0]
    check_coord = geopy.Point(check_coord[1], check_coord[0])
    for i in range(data.shape[0]):
        start = (float(data.iloc[i, 3]), float(data.iloc[i,4]))   #3 From_GPS_Lon and 4 From_GPS_Lat
        end = (float(data.iloc[i, 6]), float(data.iloc[i, 7]))   #6 To_GPS_Lon and 7 To_GPS_Lon
        bottom_left, bottom_right, top_left, top_right = find_frame_corners(start, end)
        if is_inside(check_coord, start, end):
            distress_found = True
            for coord in coords[1:]:
                coord = geopy.Point(coord[1], coord[0])
                if is_inside(coord,start,end):
                    inside_count += 1
            if inside_count == 4:
                #print(i)
                #print(list(data.iloc[i,:]))
                #print(bottom_left, bottom_right, top_left, top_right)
                return bottom_left, bottom_right, top_left, top_right, data.iloc[i, -2]
            else:
                return bottom_left, bottom_right, top_left, top_right, 'Between two frames'  # Later you need to drop this row of data since the distress box
                # is either not found or between two frames.
    if not distress_found:
        return 'NA', 'NA', 'NA', 'NA', 'NA'

print(kml_data.shape)
kml_data.to_csv('data_merged_before.csv',index=False)

for i in range(kml_data.shape[0]):
    #print(data.iloc[i, :])
    #sys.exit()
    out = find_its_image(kml_data.iloc[i,7],dff_data)
    kml_data.iloc[i, -1] = out[-1] # Image name
    kml_data.iloc[i, -2] = str(out[-2]) # top_right
    kml_data.iloc[i, -3] = str(out[-3]) # top_left
    kml_data.iloc[i, -4] = str(out[-4]) # bottom_right
    kml_data.iloc[i, -5] = str(out[-5]) # bottom_left
print(kml_data.shape)
kml_data.to_csv('data_merged_after.csv',index=False)


def draw_box(data):
    for i in range(data.shape[0]):
        print("hi")
        exit()
#print(kml_data[['ImageName']])

#data.to_csv('data_merged_before.csv',index=False)
#print(data[['ImageName']])
#sys.exit()

#data.to_csv('data_merged_after.csv',index=False)