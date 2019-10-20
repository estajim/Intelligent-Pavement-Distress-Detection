
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
                    img_name = xml_file.split("\\")[-1].split(".")[0]
                    #print(img_name)
                    #sys.exit()
                    value = (subdir, folder_name, img_name,size_array,segmented_val,distress_coords)
                    #print(value)
                    #sys.exit()
                    #value = (distress_name, distress_coords)
                xml_list.append(value)
        column_name = ['Distress Name', 'Distress Coordinates']
        column_name = ['Directory', 'Folder name','Image name','Size array','Segmented', 'Distress coordiantes']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df



image_path = os.path.join(os.getcwd(), 'DE_annotations_byKML')
kml_data = xml_to_csv(image_path)
kml_data.to_csv('training_from_KML.csv', index=None)
print('Successfully converted xml to csv.')



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
    ddf_np=np.asarray(input_lines)
    return rsp_file, ddf_np
for dff_path in glob.glob(path + '/*.dff'):  # Frame information
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
        dff_all.append(dff_data, ignore_index=True)
    except NameError:
        dff_all=dff_data
print("Size of DFF file: {}".format(dff_all.shape))
dff_data=dff_all
# ================================================= Merge KML and DFF data ======================================================
kml_data.columns = ['Directory', 'Folder name', 'SectionName', 'Size array', 'Segmented',
       'Distress coordiantes']
#print(kml_data.loc[0,'Directory'])
#print(kml_data.loc[0,'Directory'])
print(list(kml_data.columns))
print(list(dff_data.columns))
#sys.exit()
data = pd.merge(kml_data[['Directory', 'Folder name', 'SectionName', 'Size array', 'Segmented', 'Distress coordiantes']],
                 dff_data[['ImageName', 'SectionName', 'From_GPS_Lon', 'From_GPS_Lat', 'To_GPS_Lon', 'To_GPS_Lat']],
                 on='SectionName')
#print(data.head())
print("KML and DFF are merged into one unique dataframe, called data (size : {}) "
      "and the following selected columns:\n{}"
      .format(data.shape,list(data.columns)))
def is_inside(min,test,max):
    if abs(test)>= abs(min) and abs(test) <= abs(max):
        return True
    else:
        return False

def find_its_image(coords,data):
    inside_count = 0
    print(coords)
    # start with one coord and find the first match. Then check the other coords and make sure they are inside the frame range as well.
    check_coord = coords [0]
    for i in range(data.shape[0]):
        start = float(data.iloc[i,7])   #7 From_GPS_Lon and 8 From_GPS_Lat
        end = float(data.iloc[i,9])     #9 To_GPS_Lon and 10 To_GPS_Lon
        #print(start)
        #print(end)
        #print(check_coord[1])
        #print("================")
        #sys.exit()
        if is_inside(start,check_coord[1],end):
            for coord in coords[1:]:
                if is_inside(start, coord[1], end):
                    inside_count+=1
            if inside_count == 4:
                return data.iloc[i,6]
            else:
                return 'NA' # Later you need to drop this row of data since the distress box
                            # is either not found or between two frames.
data.to_csv('data_merged_before.csv',index=False)
for i in range(data.shape[0]):
    data.iloc[i,6] = find_its_image(data.iloc[i,5],data)
print(data[['ImageName']])
data.to_csv('data_merged_after.csv',index=False)
