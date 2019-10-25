from zipfile import ZipFile
import os
import glob

import xml.sax, xml.sax.handler
image_path = os.path.join(os.getcwd(), 'DE_annotations_byKML')

class PlacemarkHandler(xml.sax.handler.ContentHandler):
    def __init__(self):
        self.inName = False  # handle XML parser events
        self.inPlacemark = False
        self.mapping = {}
        self.buffer = ""
        self.name_tag = ""

    def startElement(self, name, attributes):
        if name == "Placemark":  # on start Placemark tag
            self.inPlacemark = True
            self.buffer = ""
        if self.inPlacemark:
            if name == "name":  # on start title tag
                self.inName = True  # save name text to follow

    def characters(self, data):
        if self.inPlacemark:  # on text within tag
            self.buffer += data  # save text if in title

    def endElement(self, name):
        self.buffer = self.buffer.strip('\n\t')

        if name == "Placemark":
            self.inPlacemark = False
            self.name_tag = ""  # clear current name

        elif name == "name" and self.inPlacemark:
            self.inName = False  # on end title tag
            self.name_tag = self.buffer.strip()
            self.mapping[self.name_tag] = {}
        elif self.inPlacemark:
            if name in self.mapping[self.name_tag]:
                self.mapping[self.name_tag][name] += self.buffer
            else:
                self.mapping[self.name_tag][name] = self.buffer
        self.buffer = ""




def build_table(mapping):
    sep = ','

    output = 'Name' + sep + 'Coordinates\n'
    points = ''
    lines = ''
    shapes = ''
    for key in mapping:
        coord_str = mapping[key]['coordinates'] + sep

        if 'LookAt' in mapping[key]:  # points
            points += key + sep + coord_str + "\n"
        elif 'LineString' in mapping[key]:  # lines
            lines += key + sep + coord_str + "\n"
        else:  # shapes
            shapes += key + sep + coord_str + "\n"
    output += points + lines + shapes
    return output

for xml_file in glob.glob(image_path + '/*.kml'):
    with open(xml_file, 'rb') as f:
        doc = f.read()
        parser = xml.sax.make_parser()
        handler = PlacemarkHandler()
        parser.setContentHandler(handler)
        parser.parse(doc)



        outstr = build_table(handler.mapping)
        out_filename = filename[:-3] + "csv" #output filename same as input plus .csv
        f = open(out_filename, "w")
        f.write(outstr)
        f.close()
        print(outstr)