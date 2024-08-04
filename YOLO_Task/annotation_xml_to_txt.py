import os
import xml.etree.ElementTree as ET

# Directories
input_dir = "/home/samer/Tasks/YOLO_Task/Train_Data/Final_Train_data_labeled"
output_dir = "/home/samer/Tasks/YOLO_Task/Train_Data/labels_txt"
classes = ["turbine"]  # Replace with your actual class names

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def convert_xml_to_yolo(xml_file, output_dir, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    yolo_annotations = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        if int(difficult) == 1:
            continue

        cls = obj.find('name').text
        if cls not in classes:
            continue

        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        bb = (b[0] + b[2]) / 2.0 - 1, (b[1] + b[3]) / 2.0 - 1, \
             b[2] - b[0], b[3] - b[1]
        bb = bb[0] / width, bb[1] / height, bb[2] / width, bb[3] / height
        yolo_annotations.append(f"{cls_id} {' '.join([str(a) for a in bb])}")

    # Write YOLO formatted annotation to file
    base_name = os.path.splitext(os.path.basename(xml_file))[0]
    yolo_file = os.path.join(output_dir, base_name + '.txt')
    with open(yolo_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))

def main(input_dir, output_dir, classes):
    for xml_file in os.listdir(input_dir):
        if xml_file.endswith('.xml'):
            convert_xml_to_yolo(os.path.join(input_dir, xml_file), output_dir, classes)

if __name__ == "__main__":
    main(input_dir, output_dir, classes)

