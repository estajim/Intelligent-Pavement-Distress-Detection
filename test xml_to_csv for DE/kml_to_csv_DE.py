'''
Things learned so far about treating XML files

each ElementTree object has two callers:
my_object.tag    -> which gives the tag of the object
my_object.attrib    -> which gives a dictionary attributes (if any) of the object

Attibutes are different from sub-objects within each objects. Attributes come right after opening each object.
Then each object can have multilpe sub objects which their entity is object
To access sub objects:

for sub_obj in my_object:
    print(sub_obj)
'''

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys
import math
#from pykml import parser
from fastkml import kml



def E():
    sys.exit()
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


def print_child_features(element):
    """ Prints the name of every child node of the given element, recursively """
    if not getattr(element, 'features', None):
        return
    for feature in element.features():
        print(feature.name)
        print_child_features(feature)



k = kml.KML()

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.kml'):

        with open(xml_file, 'rb') as f:
            doc = f.read()
            k = kml.KML()
            k.from_string(doc)
            #print(print_child_features(k))

            features = list(k.features())
            #print(len(features))
            #print(features[0].features())
            f2 = list(features[0].features())
            print(len(f2))
            print(f2)
            #print(f2[0])
            #print(f2[0].name)
            E()
            for item in features[0].features():
                print(item.iter())
                for itemm in item.LineString:
                    print(itemm)
            E()




        tree = ET.parse(xml_file)
        #print(tree)
        #sys.exit()
        root = tree.getroot()
        print(ET.tostring(root, encoding='utf8').decode('utf8'))
        for pm in root.iter('Placemark'):
            print(pm)
        E()

        # As a result, 'pavementData' is the root
        print("root object: {}".format(root.tag))
        #print(root.findall('inspectedElement'))
        subroot=tree.findall('Document')
        for child in root.iter('Document'):
            print(child)
        print(tree.getroot())
        print(root.attrib)
        print(subroot)
        #print(subroot.findall("inspectedElement"))

        E()
        for member in root.findall('Placemark'):
            # Member 0 is the start location
            # Member 1 is the end location
            # member[3][0] has all the distress data for each frame

            #print(member.attrib)
            filename= member.attrib['PID']+ '-' + member.attrib['inspectedElementID']
            frame_start=(float(member[0][2].attrib['x']),float(member[0][2].attrib['y']))
            frame_end = (float(member[1][2].attrib['x']), float(member[1][2].attrib['y']))
            frame_length = distance(frame_start,frame_end)
            for distress in member[3][0]:
                #print(distress.attrib["distressCode"])
                distress_code = distress.attrib["distressCode"]
                value = (filename,frame_length,distress_code)
                xml_list.append(value)

            '''
            value = (filename,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            print(value)
            xml_list.append(value)     
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    '''
    column_name = ['filename', 'length', 'Distress Code']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'DE_annotations_byKML')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('training_from_KML.csv', index=None)
    print('Successfully converted xml to csv.')


main()