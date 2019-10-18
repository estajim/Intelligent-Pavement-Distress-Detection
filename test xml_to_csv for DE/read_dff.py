import os
import sys
import numpy as np
import pandas as pd

image_path = os.path.join(os.getcwd(), 'DE_annotations_byKML')
#print(image_path)
file = 'CLEMENTINA_ST_C.dff'
file_path=os.path.join(image_path,file)
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

def load_ddf_data(file_path):
    input_lines = []
    with open(file_path,'r') as f:
        line_num=1
        for line in f:
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
    return ddf_np
print(load_ddf_data(file_path))

