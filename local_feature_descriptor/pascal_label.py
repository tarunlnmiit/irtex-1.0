from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import cv2
from tqdm import tqdm
import csv

# parse an xml file by name
# dataFile = os.getcwd()
dataFile = '/Users/tarun/Projects/dataset/VOCdevkit/VOC2012'
path = os.path.join(dataFile, 'Annotations')
data_list = [["filename", "label"]]
with open('pascal_labels.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')

    for file in tqdm(os.listdir(path)):
        tree = ET.parse((os.path.join(path, file)))
        root = tree.getroot()
        for fn in root.iter('filename'):
            filename = fn.text
        prev_name = ''
        for label in root.findall("./object/name"):
            name = label.text
            if name != prev_name:
                row = [filename, label.text]
                data_list.append(row)
                prev_name = name

    writer.writerows(data_list)
    
dataFile = '/Users/tarun/Projects/dataset/VOCdevkit/VOC2007'
path = os.path.join(dataFile, 'Annotations')
data_list = []
with open('pascal_labels.csv', 'a', newline='') as file:
    writer = csv.writer(file, delimiter=',')

    for file in tqdm(os.listdir(path)):
        tree = ET.parse((os.path.join(path, file)))
        root = tree.getroot()
        for fn in root.iter('filename'):
            filename = fn.text
        prev_name = ''
        for label in root.findall("./object/name"):
            name = label.text
            if name != prev_name:
                row = [filename, label.text]
                data_list.append(row)
                prev_name = name

    writer.writerows(data_list)