from dicttoxml import dicttoxml
import sys
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from xml.dom.minidom import parseString


dictionary = {
    'time': {"hour":"1", "minute":"30","seconds": "40"},
    'place': {"street":"40 something", "zip": "00000"}
}





import xml.etree.ElementTree as ET




def create_xml(img_path, folder_name, img_name,size_array,segmented, obj_name,obj_pose,obj_truncated,obj_dificult,obj_box_corners):
    # create the file structure
    data = ET.Element('annotation')
    folder = ET.SubElement(data,folder_name)
    filename = ET.SubElement(data, folder_name)
    path = ET.SubElement(data, img_path)
    soruce = ET.SubElement(data, "Unkown")
    size = ET.SubElement(data, folder_name)
    folder = ET.SubElement(data, folder_name)
    folder = ET.SubElement(data, folder_name)

    items = ET.SubElement(data, 'items')
    item1 = ET.SubElement(items, 'item')
    item2 = ET.SubElement(items, 'item')
    item1.set('name', 'item1')
    item2.set('name', 'item2')
    item1.text = 'item1abc'
    item2.text = 'item2abc'


    # create a new XML file with the results
    mydata = ET.tostring(data)
    myfile = open("items2.xml", "wb")
    myfile.write(mydata)
    myfile

    my_dict={
        "folder":folder_name, "filename":img_name, "path": path, "soruce":{"database":"Unkown"},
        "size": {"width":size_array[0],
                 "height":size_array[1],
                 "depth":size_array[2]},
        "segmented":segmented

    }
xml = dicttoxml(dictionary, custom_root='test', attr_type=False)

dom = parseString(xml)
print(dom.toprettyxml())