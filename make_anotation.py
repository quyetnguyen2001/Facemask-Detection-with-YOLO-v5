import os 
import xml.etree.ElementTree as ET
import cv2
path = "data/dataset/annotations"

files = os.listdir(path)
classes = ["without_mask" , "with_mask" , "mask_weared_incorrect"]

for file_ in files:
    file = os.path.join(path, file_)
    tree = ET.parse(file)
    root = tree.getroot()
    lines = ''
    file_name = root.find("filename").text
    
    
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    for obj in root.findall("object"):
        label = classes.index(obj.find("name").text)
        x1, y1, x2, y2 = [int(float(obj.find('bndbox').find(tag).text)) - 1 for tag in
                          ["xmin", "ymin", "xmax", "ymax"]]
        x = ((x1+x2)/2)/width
        y = ((y1+y2)/2)/height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        line = f'{label} {x} {y} {w} {h} \n'
        
        lines+=line
        


    # with open(f"data/dataset/labels/{file_name.replace('.jpg','.txt').replace('.png','.txt')}",'w+') as f:
    #     f.write(lines)