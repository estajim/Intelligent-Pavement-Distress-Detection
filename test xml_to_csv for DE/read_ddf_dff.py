import os
import sys
import numpy as np
import pandas as pd
import math

image_path = os.path.join(os.getcwd(), 'DE_annotations_byKML')
#print(image_path)
file = 'CLEMENTINA_ST_C'
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
ddf_file = file_path+'.ddf'
dff_file = file_path+'.dff'
ddf_np = load_data(ddf_file)[1]
RSP_name = load_data(ddf_file)[0]
dff_np = load_data(dff_file)[1]

# Load Pandas Dataframes for ddf and dff files
ddf_data = pd.DataFrame(ddf_np[1:,:],columns=ddf_np[0,:],index=None)
dff_data = pd.DataFrame(dff_np[1:,:],columns=dff_np[0,:],index=None)
print("Size of DDF file: {}".format(ddf_np.shape))
print("Size of DFF file: {}".format(dff_np.shape))

# Preparations for Image file names in the HDC folder using DDF file
ddf_data = ddf_data.astype({'From(ft)': 'float32','To(ft)':'float32'})
ddf_data[['From(ft)','To(ft)']] = ddf_data[['From(ft)','To(ft)']]/20
ddf_data = ddf_data.astype({'From(ft)': 'int32','To(ft)':'int32'})
ddf_data[['From(ft)','To(ft)']] = ddf_data[['From(ft)','To(ft)']]*20.0
#print(ddf_data[['From(ft)','To(ft)']])
ddf_data = ddf_data.astype({'From(ft)': 'str','To(ft)':'str'})
ddf_data['ImageName']=RSP_name+'      '+ ddf_data[['From(ft)']]+'_Image3D'
#print(ddf_data['ImageName'])


# Preparations for Image file names in the HDC folder using DFF file
dff_data = dff_data.astype({'From_Station': 'float32','To_Station':'float32'})
dff_data[['From_Station','To_Station']] = dff_data[['From_Station','To_Station']]/20
dff_data = dff_data.astype({'From_Station': 'int32','To_Station':'int32'})
dff_data[['From_Station','To_Station']] = dff_data[['From_Station','To_Station']]*20.0
#print(ddf_data[['From(ft)','To(ft)']])
dff_data = dff_data.astype({'From_Station': 'str','To_Station':'str'})
dff_data['ImageName']=RSP_name+'      '+ dff_data[['From_Station']]+'_Image3D'
#print(dff_data['ImageName'])
#print(ddf_data.columns)
# Merge DDF and DFF files with shared ImageName columns
data = pd.merge(ddf_data[['ImageName','PID', 'DistressType', 'MPID', 'Severity', 'Width(ft)', 'Length(ft)','From(ft)', 'To(ft)', 'Latitude', 'Longitude']],
                 dff_data[['ImageName', 'From_GPS_Lon', 'From_GPS_Lat', 'To_GPS_Lon', 'To_GPS_Lat']],
                 on='ImageName')
#print(data.head())
print("DDF and DFF are merged into one unique dataframe, called data (size : {}) "
      "and the following selected columns:\n{}"
      .format(data.shape,list(data.columns)))


def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371000  # m
    radius = radius*0.000621371*5280 # ft

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

def calc_bearing_NG(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def calc_bearing(pointA, pointB):
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    lon1 = math.radians(pointA[1])
    lon2 = math.radians(pointB[1])

    y = math.sin(lon2-lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(lon2-lon1)
    bearing  = math.degrees(math.atan2(y, x))
    return bearing


print()
#print(math.sin(math.radians(30)))
'''
(all calculations in ft)

theta = difference in bearing of (start,end) and (start, distress_Center)
xmin = b/2 - cos(90-theta)
ymin = sin(90-theta)

xmax = xmin + distress_box_width
ymax = ymin + distress_box_height

box_corners = (xmin, ymin, xmax, ymax)
'''
def find_corners(data):
    corners = []
    for i in range(data.shape[0]):
        p = (float(data.loc[i, 'Latitude']), float(data.loc[i, 'Longitude']))
        start = (float(data.loc[i, 'From_GPS_Lon']), float(data.loc[i, 'From_GPS_Lat']))
        end = (float(data.loc[i, 'To_GPS_Lon']), float(data.loc[i, 'To_GPS_Lat']))
        w_frame = 13.64173
        h_frame = 20
        w_distress = float(data.loc[0, 'Width(ft)'])
        h_distress = float(data.loc[0, 'Length(ft)'])

        # print(calc_ancgle(start,p))
        # print(calc_ancgle(start,end))
        theta = 90 - (calc_bearing(start, p) - calc_bearing(start, end))
        # print(theta)

        # sys.exit()
        # print(data.loc[0,'DistressType'])
        # print(data.loc[0,'Latitude'])
        # print(data.loc[0,'To(ft)'])
        # sys.exit()

        xmin = w_frame / 2 - math.cos(math.radians(theta)) * distance(start, p) - w_distress / 2
        xmax = xmin + w_distress
        ymin = math.sin(math.radians(theta)) * distance(start, p) - h_distress / 2
        ymax = ymin + h_distress

        corners.append((xmin, ymin, xmax, ymax))
    return corners
def find_distress_box(row):
    for i in range(data.shape[0]):
        p = (float(row['Latitude']), float(row['Longitude']))
        start = (float(row['From_GPS_Lon']), float(row['From_GPS_Lat']))
        end = (float(row['To_GPS_Lon']), float(row['To_GPS_Lat']))
        w_frame = 13.64173
        h_frame = 20
        w_distress = float(row['Width(ft)'])
        h_distress = float(row['Length(ft)'])
        theta = 90 - (calc_bearing(start, p) - calc_bearing(start, end))

        xmin = w_frame / 2 - math.cos(math.radians(theta)) * distance(start, p) - w_distress / 2
        xmax = xmin + w_distress
        ymin = math.sin(math.radians(theta)) * distance(start, p) - h_distress / 2
        ymax = ymin + h_distress

    return (round(xmin,3), round(ymin,3), round(xmax,3), round(ymax,3))
data = data [data['MPID']=='5']  # only take patching
data['Distress_box_coordinates'] = np.zeros((data.shape[0],))
corners=[]
for i in range(data.shape[0]):
    row = data.iloc[i, :]
    print(find_distress_box(row))
    corners.append(find_distress_box(row))
#print(data[['Latitude','Longitude','From_GPS_Lon','From_GPS_Lat']])
#print(data.shape)
#print(corners)

data['Distress_box_coordinates'] = corners
print(data[['ImageName','DistressType','Distress_box_coordinates']])
sample_out = data[['ImageName','DistressType','Distress_box_coordinates']]
sample_out.to_csv('sample_out2.csv',index=False)
