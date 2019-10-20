
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys
import math
#from pykml import parser
from fastkml import kml
import xml.etree.ElementTree as ET


k = kml.KML()
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


def xml_to_csv(path):

    xml_list = []
    for subdir, dirs, files in os.walk(path):
        for xml_file in glob.glob(subdir + '/*.kml'):
            print(xml_file)
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
                    size_array=(200,500,3)
                    segmented_val = 0
                    img_name = glob.glob(subdir + '/*.kml')[0].split("\\")[-1].split(".")[0]
                    #print(img_name)
                    value = (subdir, folder_name, img_name,size_array,segmented_val,distress_coords)
                    #print(value)
                    #sys.exit()
                    #value = (distress_name, distress_coords)
                xml_list.append(value)
        column_name = ['Distress Name', 'Distress Coordinates']
        column_name = ['Directory', 'Folder name','Image name','Size array','Segmented', 'Distress coordiantes']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'DE_annotations_byKML')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('training_from_KML.csv', index=None)
    print('Successfully converted xml to csv.')


main()