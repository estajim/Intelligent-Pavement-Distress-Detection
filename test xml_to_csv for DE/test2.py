
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

def xml_to_csv(path):

    xml_list = []
    for xml_file in glob.glob(path + '/*.kml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        lineStrings = tree.findall('.//{http://www.opengis.net/kml/2.2}LineString')
        Placemarks = tree.findall('.//{http://www.opengis.net/kml/2.2}Placemark')
        print(len(lineStrings))
        print(len(Placemarks))

        for placemark, LineString in zip(Placemarks,lineStrings):
            for placemark_features in placemark:
                if placemark_features.tag == '{http://www.opengis.net/kml/2.2}name':
                    distress_name = placemark_features.text
            #print(LineString)
            for LineString_feature in LineString:
                distress_coords=list()
                if LineString_feature.tag == '{http://www.opengis.net/kml/2.2}coordinates':
                    #print(LineString_feature.tag, LineString_feature.text)
                    coords_list = LineString_feature.text.replace("\n",',').split(',')[1:-1]
                    print(coords_list)
                    i=0
                    while i<len(coords_list):
                        distress_coords.append((float(coords_list[i]),float(coords_list[i+1])))
                        i+=2
                    print(distress_coords)
                    #sys.exit()
                value = (distress_name,distress_coords)
            xml_list.append(value)
    column_name = ['Distress Name', 'Distress Coordinates']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'DE_annotations_byKML')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('training_from_KML.csv', index=None)
    print('Successfully converted xml to csv.')


main()